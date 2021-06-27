import afssh as fssh
from datetime import date
import time
import numpy as np
import math as math
import boltzmann_sampling as sampling
from multiprocessing import Pool
import os


class Batch:
    def __init__(self, stopping_function):
        self.stopping_function = stopping_function

        self.batch_state = "initiated"
        self.batch_error = None

        self.start_time = None
        self.end_time = None

        self.states = []

    def run(self, fssh_settings, num_particles=10):
        self.model = fssh_settings.get("model")
        self.m = fssh_settings.get("mass")
        self.v0 = fssh_settings.get("v0")
        self.k = self.m*(math.sqrt(np.sum(self.v0**2)))
        self.dt_c = fssh_settings.get("dt_c")
        self.max_iter = fssh_settings.get("max_iter")
        self.r0 = fssh_settings.get("r0")
        self.debug = fssh_settings.get("debug")
        self.langevin = fssh_settings.get("langevin")
        self.deco = fssh_settings.get("deco")
        self.e_tol = fssh_settings.get("e_tol")
        self.coeff = fssh_settings.get("coeff")
        self.t0 = fssh_settings.get("t0")
        self.state0 = fssh_settings.get("state0")
        self.seed = fssh_settings.get("seed")

        self.num_particles = num_particles

        try:
            self.batch_state = "finished"
            self.start_time = time.time()
            for i in range(num_particles):
                print(i+1, "/", num_particles)
                x = fssh.AFSSH(self.model, self.r0, self.v0, self.dt_c, self.e_tol, self.coeff,
                               self.m, self.t0, self.state0, self.deco, self.langevin, self.seed)
                x.run(self.max_iter, self.stopping_function, self.debug)

                self.states.append(
                    (x.r, x.v, x.lam, x.t, x.coeff, x.i, x.switches))
        except Exception as e:
            self.batch_state = "failed"
            self.batch_error = e
        finally:
            self.end_time = time.time()

    def generate_report(self, outfile):
        outfile += ".txt"
        with open(outfile, 'w') as f:
            self.write_heading(f)
            self.enumerate_states(f)

    def write_heading(self, f):
        lines = []
        lines.append(date.today().isoformat())
        lines.append(f"\nJob state: {self.batch_state}\n")
        lines.append(f"Potential model: {type(self.model).__name__}\n")
        lines.append(
            f"Job time: {self.end_time - self.start_time} seconds\n")

        if self.batch_state == "failed":
            lines.append("Job failed...\n")
            lines.append(str(self.batch_error) + "\n")
            f.writelines(lines)
        elif self.batch_state == "initiated":
            lines.append("Job has not been run...\n")
            f.writelines(lines)
        else:
            lines.append(("-"*10) + "Job parameters" + ("-"*10) + "\n")
            lines.append(f"Num particles: {self.num_particles}\n")
            lines.append(f"Max iter: {self.max_iter}\n")
            lines.append(f"Time step: {self.dt_c}\n")
            lines.append(f"Particle momentum: {self.k}\n")
            lines.append(f"Start position: {self.r0}\n")
            lines.append(f"Particle mass: {self.m}\n")
            lines.append(f"Particle velocity: {self.v0}\n")
            lines.append(f"Start coefficients: {self.coeff}\n")
            lines.append(f"Start state: {self.state0}\n")
            lines.append(f"Seed: {self.seed}\n")

            if self.deco != None:
                lines.append(f"Decoherence accounted for: \n")
                lines.append(f"Start delta_R: {self.deco.get('delta_R')}\n")
                lines.append(f"Start delta_P: {self.deco.get('delta_P')}\n")
            else:
                lines.append("Decoherence calculations turned off\n")

            if self.langevin != None:
                lines.append(f"Langevin dynamics accounted for: \n")
                lines.append(
                    f"Damping coefficient: {self.langevin.get('damp')}\n")
                lines.append(f"Temperature: {self.langevin.get('temp')}\n")
            else:
                lines.append("Langevin dynamics turned off\n")

            lines.append(("-"*10) + "Job results" + ("-"*10) + '\n')
            f.writelines(lines)

    def enumerate_states(self, f):
        avg_pos = 0
        avg_state = 0
        avg_v = 0
        lines = []

        lines.append(("-"*10) + "Particle results" + ("-"*10) + "\n")
        for i in range(len(self.states)):
            state = self.states[i]
            lines.append(str(i) + "\n")
            lines.append(f"position: {state[0]}\n")
            lines.append(f"velocity: {state[1]}\n")
            lines.append(f"electronic state: {state[2]}\n")
            lines.append(f"end_time: {self.dt_c*state[5]}\n")
            lines.append(
                f"end electronic coefficients: {state[4]}\n")
            lines.append("State switches:\n")
            for switch in state[6]:
                line = "" if switch.get("success") else "failed "
                line += f"state switch {switch.get('old_state')} "
                line += "-> " if switch.get("success") else "-/> "
                line += f"{switch.get('new_state')}. coeff: {switch.get('coefficients')}, "
                line += f"deltaV: {switch.get('delta_v')}, "
                line += f"velocity: {switch.get('velocity')}, "
                line += f"position: {switch.get('position')}\n"
                lines.append(line)

            avg_pos += state[0]
            avg_v += state[1]
            avg_state += state[2]

        avg_pos /= self.num_particles
        avg_v /= self.num_particles
        avg_state /= self.num_particles

        f.write(f"Avg position: {avg_pos}\n")
        f.write(f"Avg velocity: {avg_v}\n")
        f.write(f"Avg electronic state: {avg_state}\n")
        f.writelines(lines)
        return

    def output_csv(self, outfile):
        outfile += ".csv"
        with open(outfile, 'w') as f:
            f.write("position,velocity,electronic_state,coefficient\n")
            for state in self.states:
                f.write(f"{state[0]},{state[1]},{state[2]},{state[4]}\n")


class New_Batch():
    def __init__(self, fssh_settings, stopping_fcn):
        self.model = fssh_settings.get("model")
        self.m = fssh_settings.get("mass")
        self.v0 = fssh_settings.get("v0")
        self.k = self.m*(math.sqrt(np.sum(self.v0**2)))
        self.dt_c = fssh_settings.get("dt_c")
        self.max_iter = fssh_settings.get("max_iter")
        self.r0 = fssh_settings.get("r0")
        self.debug = fssh_settings.get("debug")
        self.langevin = fssh_settings.get("langevin")
        self.deco = fssh_settings.get("deco")
        self.e_tol = fssh_settings.get("e_tol")
        self.coeff = fssh_settings.get("coeff")
        self.t0 = fssh_settings.get("t0")
        self.state0 = fssh_settings.get("state0")

        self.stopping_fcn = stopping_fcn
        self.dim = len(self.r0) if hasattr(self.r0, "__len__") else 1

        self.batch_state = "initiated"
        self.batch_error = None
        self.start_time = None
        self.end_time = None

    def run(self, num_particles, outfile, outfolder, num_cores=1, boltzmann_vel=False, temp=None, verbose=False, seeds=None):
        self.num_particles = num_particles
        self.boltzmann_vel = boltzmann_vel
        self.temp = temp

        with open(outfile, "w") as f:
            self.write_heading(f)

        try:
            self.batch_state = "finished"
            self.start_time = time.time()
            if boltzmann_vel:
                v = sampling.v_sample(temp, self.m, (num_particles, self.dim))
            else:
                v = np.zeros((num_particles, self.dim)) + self.v0

            num_loops = int(num_particles/num_cores)

            for i in range(num_loops):
                if num_cores > 1:
                    pool = Pool()
                    results = []
                    # Run async
                    for j in range(num_cores):
                        index = (i*num_cores) + j
                        temp = f"{index}.tmp"
                        args = [index, v[index], temp,
                                seeds[index], verbose, outfolder]
                        results.append(pool.apply_async(self.run_traj, args))

                    # Wait until all are done
                    for j in range(num_cores):
                        results[j].wait()

                    # Collate temp files into main outfile
                    for j in range(num_cores):
                        index = (i*num_cores) + j
                        temp = f"{index}.tmp"
                        with open(outfile, 'a') as out, open(f"{index}.tmp", "r") as infile:
                            out.writelines(infile.readlines())
                        os.remove(temp)

                else:
                    self.run_traj(
                        index, v[index], outfile, seed=seeds[index], verbose=verbose, outfolder=outfolder)

        except Exception as e:
            print(e.__traceback__.tb_lineno)
            self.batch_state = "failed"
            self.batch_error = e.__str__()
        finally:
            self.end_time = time.time()
            with open(outfile, "a") as f:
                lines = []
                if self.batch_state == "finished":
                    lines.append("Job completed successfully\n")
                else:
                    lines.append("Job failed to run\n")
                    lines.append(self.batch_error + "\n")
                lines.append(
                    f"Time to finish: {self.end_time - self.start_time} seconds\n")
                f.writelines(lines)

    def write_heading(self, f):
        lines = []
        lines.append(date.today().isoformat())
        lines.append(f"\nJob state: {self.batch_state}\n")
        lines.append(f"Potential model: {type(self.model).__name__}\n")
        lines.append(("-"*10) + "Job parameters" + ("-"*10) + "\n")
        lines.append(f"Num particles: {self.num_particles}\n")
        lines.append(f"Max iter: {self.max_iter}\n")
        lines.append(f"Time step: {self.dt_c}\n")
        lines.append(f"Particle momentum: {self.k}\n")
        lines.append(f"Start position: {self.r0}\n")
        lines.append(f"Particle mass: {self.m}\n")
        lines.append(f"Particle velocity: {self.v0}\n")
        lines.append(f"Start coefficients: {self.coeff}\n")
        lines.append(f"Start state: {self.state0}\n")

        if self.deco != None:
            lines.append(f"Decoherence accounted for: \n")
            lines.append(f"Start delta_R: {self.deco.get('delta_R')}\n")
            lines.append(f"Start delta_P: {self.deco.get('delta_P')}\n")
        else:
            lines.append("Decoherence calculations turned off\n")

        if self.langevin != None:
            lines.append(f"Langevin dynamics accounted for: \n")
            lines.append(
                f"Damping coefficient: {self.langevin.get('damp')}\n")
            lines.append(f"Temperature: {self.langevin.get('temp')}\n")
        else:
            lines.append("Langevin dynamics turned off\n")

        if self.boltzmann_vel:
            lines.append(
                f"Initializing velocities from Boltzmann Distribution\n")

        lines.append(("-"*10) + "Job results" + ("-"*10) + '\n')
        f.writelines(lines)

    def log_step(self, fssh, f):
        if fssh.i % 25 == 0:
            np.savetxt(f, fssh.r, newline="|")
            f.write(",")
            np.savetxt(f, fssh.v, newline="|")
            f.write(",")
            np.savetxt(f, fssh.coeff, newline="|")
            f.write(f",{fssh.lam}\n")

    def log_trajectory(self, fssh, f, i, traj_time):
        lines = []
        lines.append(str(i) + "\n")
        lines.append(f"Start velocity: {fssh.v0}\n")
        lines.append(f"End position: {fssh.r}\n")
        lines.append(f"End velocity: {fssh.v}\n")
        lines.append(f"End coefficients: {fssh.coeff}\n")
        lines.append(f"End KE: {fssh.calc_KE(fssh.v)}\n")
        lines.append(f"End state: {fssh.lam}\n")
        lines.append(f"End time: {fssh.t}\n")
        lines.append(f"Seed: {fssh.seed}\n")
        lines.append(
            f"Time to end: {traj_time} seconds\n")
        lines.append("===State switches===\n")
        for s in fssh.switches:
            lines.append(f"old_state: {s.get('old_state')}, ")
            lines.append(f"new_state: {s.get('new_state')}, ")
            lines.append(f"position: {s.get('position')}, ")
            lines.append(f"velocity: {s.get('velocity')}, ")
            lines.append(f"coefficients: {s.get('coefficients')}, ")
            lines.append(f"delta_v: {s.get('delta_v')}, ")
            lines.append(f"success: {s.get('success')}\n")
        lines.append("===End state switches===\n")
        f.writelines(lines)

    def run_traj(self, i, v, outfile, seed=None, verbose=False, outfolder=None):
        t_start = time.time()
        print(i)
        x = fssh.AFSSH(self.model, self.r0, v, self.dt_c, self.e_tol, self.coeff,
                       self.m, self.t0, self.state0, self.deco, self.langevin, seed)
        if verbose:
            with open(outfolder + f"/{i}.csv", 'w') as f:
                def callback(fssh): return self.log_step(fssh, f)
                x.run(self.max_iter, self.stopping_fcn,
                      self.debug, callback)
        else:
            x.run(self.max_iter, self.stopping_fcn, self.debug)

        t_end = time.time()
        with open(outfile, 'a') as f:
            self.log_trajectory(x, f, i, t_end-t_start)

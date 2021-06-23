import afssh as fssh
from datetime import date
import time
import numpy as np
import math as math


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


class Distribution_Batch():
    def __init__(self, stopping_function):
        pass

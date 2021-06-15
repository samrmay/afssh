import afssh as fssh
from datetime import date
import time
import numpy as np
import math as math


class Batch:
    def __init__(self, model, stopping_function, cat_function):
        self.model = model
        self.stopping_function = stopping_function
        self.cat_function = cat_function

        self.batch_state = "initiated"
        self.batch_error = None

        self.start_time = time.time()
        self.end_time = self.start_time

        self.states = []

    def run(self, num_particles=10, start_x=-10, v=.015, max_iter=5000, debug=False, mass=2000, del_t=.5):
        self.k = mass*(math.sqrt(np.sum(v**2)))
        self.v = v
        self.m = mass
        self.del_t = del_t
        self.max_iter = max_iter
        self.num_particles = num_particles
        self.start_x = start_x
        self.debug = debug

        try:
            self.batch_state = "finished"
            for i in range(num_particles):
                print(i+1, "/", num_particles)
                x = fssh.AFSSH(self.model, start_x, self.v, del_t, mass=mass)
                x.run(max_iter, self.stopping_function, self.debug)

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
            lines.append(str(self.batch_error))
            f.writelines(lines)
        elif self.batch_state == "initiated":
            lines.append("Job has not been run...\n")
            f.writelines(lines)
        else:
            lines.append(("-"*10) + "Job parameters" + ("-"*10) + "\n")
            lines.append(f"Num particles: {self.num_particles}\n")
            lines.append(f"Max iter: {self.max_iter}\n")
            lines.append(f"Time step: {self.del_t}\n")
            lines.append(f"Particle momentum: {self.k}\n")
            lines.append(f"Start position: {self.start_x}\n")
            lines.append(f"Particle mass: {self.m}\n")
            lines.append(f"Particle velocity: {self.v}\n")

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
            lines.append(f"end_time: {self.del_t*state[5]}\n")
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
                line += f"position: {switch.get('position')}"
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

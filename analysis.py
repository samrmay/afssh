import numpy as np
import matplotlib.pyplot as plt
import os
import models as m


def plot_trajectories(ax, m, infolder):
    filenames = [f for f in os.listdir(
        infolder) if os.path.isfile(os.path.join(infolder, f))]
    files = [infolder + f for f in filenames]
    positions = []
    surfaces = []
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            pos = np.zeros(len(lines))
            sur = np.zeros(len(lines))
            for i in range(len(lines)):
                line = lines[i]
                arr = line.split(",")
                pos[i] = float(arr[0][:-1])
                sur[i] = int(arr[-1][:-1])
            positions.append(pos)
            surfaces.append(sur)

    for i in range(len(positions)):
        pos = positions[i]
        sur = surfaces[i]
        energies = np.zeros(len(pos))
        for j in range(len(pos)):
            energies[j] = m.get_adiabatic_energy(pos[j])[int(sur[j])]

        ax.plot(pos, energies)


fig, ax = plt.subplots()
ax.set_ylabel('Potential (Eh)')
ax.set_xlabel('Nuclear coordinate (a.u.)')
model = m.NState_Spin_Boson(l_states=10, r_states=10)
model = m.NState_Spin_Boson(l_states=10, r_states=10)
x = np.linspace(-20, 20, 1000)
m.plot_1d(ax, model, x)
plot_trajectories(ax, model, "results/N_state_test_062621/verbose/")
plt.show()

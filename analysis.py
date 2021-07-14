import numpy as np
import matplotlib.pyplot as plt
import os
import models as m


def read_pos_sur(infolder):
    filenames = [f for f in os.listdir(
        infolder) if os.path.isfile(os.path.join(infolder, f))]
    files = [infolder + f for f in filenames]
    positions = []
    surfaces = []
    for file in files:
        try:
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
        except:
            print("Bad csv: ", file)

    return positions, surfaces


def plot_trajectories(ax, m, infolder):
    positions, surfaces = read_pos_sur(infolder)

    for i in range(len(positions)):
        pos = positions[i]
        sur = surfaces[i]
        energies = np.zeros(len(pos))
        for j in range(len(pos)):
            energies[j] = m.get_adiabatic_energy(pos[j])[int(sur[j])]

        ax.plot(pos, energies)


def avg_trajectories(ax, m, infolder):
    positions, surfaces = read_pos_sur(infolder)
    energies = []
    for i in range(len(positions)):
        pos = positions[i]
        sur = surfaces[i]
        en = np.zeros(len(pos))
        for j in range(len(pos)):
            en[j] = m.get_adiabatic_energy(pos[j])[int(sur[j])]
        energies.append(en)

    max_len = len(max(positions, key=lambda x: len(x)))
    for i in range(len(positions)):
        l = len(positions[i])
        if l < max_len:
            positions[i] = np.concatenate((
                positions[i], np.zeros(max_len - l) + positions[i][-1]))
            energies[i] = np.concatenate(
                (energies[i], np.zeros(max_len - l) + energies[i][-1]))

    p_avg = np.zeros(max_len)
    e_avg = np.zeros(max_len)
    for i in range(len(positions)):
        p_avg += positions[i]
        e_avg += energies[i]

    p_avg /= len(positions)
    e_avg /= len(positions)

    ax.plot(p_avg, e_avg)


def outfile_analysis(outfile):
    with open(outfile, "r") as f:
        end_states = []
        end_times = []
        for l in f.readlines():
            if "end state:" in l.lower():
                end_states.append(int(l.rstrip()[-1]))
            elif "end time:" in l.lower():
                end_times.append(int(l.split(":")[1].rstrip()))

    print("Avg end state: ", sum(end_states)/len(end_states))
    print("Avg end time: ", sum(end_times)/len(end_times))


def time_spent(infolder, x_split=0):
    positions, surfaces = read_pos_sur(infolder)
    left_count = 0
    right_count = 0
    state_counts = {}
    for i in range(len(positions)):
        pos = positions[i]
        sur = surfaces[i]
        l = len(pos)
        left = sum(np.less(pos, np.zeros(l) + x_split))
        right = l - left
        left_count += left
        right_count += right
        for s in sur:
            if s in state_counts.keys():
                state_counts[s] += 1
            else:
                state_counts[s] = 1

    tot = left_count + right_count
    print("left: ", left_count/tot)
    print("right: ", right_count/tot)

    s_counts = np.array(list(state_counts.values()))
    print("state counts: ", s_counts/sum(s_counts))


def traj():
    fig, ax = plt.subplots()
    ax.set_ylabel('Potential (Eh)')
    ax.set_xlabel('Nuclear coordinate (a.u.)')
    model = m.NState_Spin_Boson(l_states=10, r_states=10)
    x = np.linspace(-20, 20, 1000)
    m.plot_1d(ax, model, x)
    # plot_trajectories(ax, model, "results/Nstate_063021/verbose/")
    avg_trajectories(ax, model, "results/Nstate_063021/verbose/")
    plt.show()


def tully_props(ax1, ax2, ax3, infolder, k_arr):
    filenames = [f"{k}.out" for k in k_arr]
    files = [infolder + f for f in filenames]

    dat = {}

    for i in range(len(files)):
        filename = files[i]
        k = k_arr[i]
        state0_reflected = 0
        state0_transmitted = 0
        state1_transmitted = 0
        with open(filename, "r") as f:
            lines = f.readlines()
            end_pos = None
            end_state = None
            for line in lines:
                if "end position:" in line.lower():
                    if end_pos is not None:
                        if end_pos > 0 and end_state == 0:
                            state0_transmitted += 1
                        elif end_pos > 0 and end_state == 1:
                            state1_transmitted += 1
                        else:
                            state0_reflected += 1

                    end_pos = float(line.split("[")[1].rstrip()[:-1])
                elif "end state:" in line.lower():
                    end_state = int(line.split(":")[1].rstrip()[-1])

        tot = state0_reflected + state0_transmitted + state1_transmitted
        state0_transmitted /= tot
        state0_reflected /= tot
        state1_transmitted /= tot

        dat[k] = (state0_transmitted, state0_reflected, state1_transmitted)

    s0_t = np.array([dat[k][0] for k in k_arr])
    s0_r = np.array([dat[k][1] for k in k_arr])
    s1_t = np.array([dat[k][2] for k in k_arr])

    ax1.plot(k_arr, s0_t)
    ax2.plot(k_arr, s0_r)
    ax3.plot(k_arr, s1_t)


def filter_finished(d):
    finished_trajs = []
    filenames = [f for f in os.listdir(d) if f.endswith(".tmp")]
    for f in filenames:
        i = int(f.split(".")[0])
        finished_trajs.append(i)
    for f in os.listdir(d + "verbose/"):
        if not int(f.split(".")[0]) in finished_trajs:
            os.remove(d + "verbose/" + f)


def plot_end_pos(outfile, m, ax):
    with open(outfile, "r") as f:
        end_pos = []
        end_state = []
        for l in f.readlines():
            if "end position:" in l.lower():
                end_pos.append(float(l.split("[")[1].rstrip()[:-1]))
            elif "end state:" in l.lower():
                end_state.append(int(l.split(":")[1].rstrip()[-1]))

    end_pos = np.array(end_pos)
    end_state = np.array(end_state)

    for i in range(len(end_pos)):
        pos = end_pos[i]
        sur = end_state[i]
        energy = m.get_adiabatic_energy(pos)[sur]
        ax.plot(pos, energy, 'ro')


def combine_tmp(d, outfile):
    filenames = [f for f in os.listdir(d) if f.endswith(".tmp")]
    for filename in filenames:
        with open(d + outfile, "a") as out, open(d + filename) as infile:
            out.writelines(infile.readlines())


def del_tmp(d):
    filenames = [f for f in os.listdir(d) if f.endswith(".tmp")]
    for filename in filenames:
        os.remove(os.path.join(d, filename))


def prepare_unfinished_job(d, out):
    filter_finished(d)
    combine_tmp(d, out)
    del_tmp(d)

# prepare_unfinished_job(
#     "results/070821_Nstate_high_density/", "070821_high_d.out")
# outfile_analysis("results/Nstate_063021/063021.out")
# time_spent("./results/Nstate_063021/verbose/")


fig, ax = plt.subplots()
ax.set_xlabel("Nuclear coordinate (au)")
ax.set_ylabel("Potential (Eh)")
# ax.set_title("iter=12500, damp=.0001, T=298")
x = np.linspace(-20, 20, 1000)
model = m.NState_Spin_Boson(l_states=10, r_states=10)
m.plot_diabats_1d(ax, model, x)
avg_trajectories(ax, model, "results/070621_long_test/verbose/")

plot_end_pos("results/070621_long_test/070621.out", model, ax)
plt.show()

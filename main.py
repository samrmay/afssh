import batch as batch
import models as models
import argparse
import re
import numpy as np
import stopping_functions as fcns


def parse_infile(inpath):
    with open(inpath, "r") as f:
        lines = f.read()

    settings = {
        "model": models.Simple_Avoided_Crossing(),
        "mass": 1,
        "v0": 0,
        "r0": 0,
        "dt_c": 20,
        "max_iter": 1000,
        "debug": False,
        "langevin": None,
        "deco": None,
        "e_tol": 1e-3,
        "coeff": None,
        "t0": 0,
        "state0": 0
    }

    b_settings = {
        "num_particles": 100,
        "boltzmann_vel": False,
        "temp": None,
        "seeds": None,
        "num_cores": 1,
        "verbose": False
    }

    flags = lines.split("%")
    if "" in flags:
        flags.remove("")

    for i in range(len(flags)):
        flag = flags[i].rstrip("\n").lower()
        flag = re.split(re.compile(r"\s+"), flag)
        arg = flag[0]
        if arg == "model":
            model_name = flag[1]
            # Split into keyword arg lists based on end positions
            keyword_args = {}
            key = None
            arr = []

            for item in flag[2:]:
                if item == "end":
                    if key != None:
                        keyword_args[key] = np.array(arr)
                        key = None
                        arr = []
                elif key == None:
                    key = item
                else:
                    arr += [float(item)]

            # Pass keyword dict into model when instantiating
            if model_name == "tully1":
                m = models.Simple_Avoided_Crossing(**keyword_args)
            elif model_name == "tully2":
                m = models.Double_Avoided_Crossing(**keyword_args)
            elif model_name == "tully3":
                m = models.Extended_Coupling_With_Reflection(**keyword_args)
            elif model_name == "nspinboson":
                m = models.NState_Spin_Boson(**keyword_args)

            settings["model"] = m
        elif arg == "mass":
            settings["mass"] = float(flag[1])
        elif arg == "velocity":
            v = []
            for dim in range(1, len(flag) - 1):
                if flag[dim] == "end":
                    break
                v.append(float(flag[dim]))
            settings["v0"] = np.array(v, dtype=float)
        elif arg == "pos" or arg == "position":
            r = []
            for dim in range(1, len(flag) - 1):
                if flag[dim] == "end":
                    break
                r.append(float(flag[dim]))
            settings["r0"] = np.array(r, dtype=float)
        elif arg == "dt_c":
            settings["dt_c"] = int(flag[1])
        elif arg == "max_iter" or arg == "iter":
            settings["max_iter"] = int(flag[1])
        elif arg == "debug":
            settings["debug"] = bool(int(flag[1]))
        elif arg == "langevin":
            lan = {
                "temp": 298.15,
                "damp": 1e-4
            }
            for i in range(len(flag)):
                item = flag[i]
                if item == "temp":
                    lan["temp"] = float(flag[i+1])
                if item == "damp":
                    lan["damp"] = float(flag[i+1])

            settings["langevin"] = lan
        elif arg == "deco":
            deco = {
                "delta_R": 0,
                "delta_P": 0
            }
            key = None
            arr = []
            for item in flag:
                if item == "delta_r":
                    key = "delta_R"
                elif item == "delta_p":
                    key = "delta_P"
                elif item == "end":
                    if key != None:
                        deco[key] = np.array(arr, dtype=float)
                        key = None
                        arr = []
                else:
                    arr += [float(item)]

            settings["deco"] = deco
        elif arg == "e_tol" or arg == "tolerance":
            settings["e_tol"] = float(flag[1])
        elif arg == "coeff":
            coeff = []
            for state in range(1, len(flag) - 1):
                if flag[dim] == "end":
                    break
                coeff.append(float(flag[state]))
            settings["coeff"] = np.array(coeff, dtype=complex)
        elif arg == "t0":
            settings["t0"] = float(flag[1])
        elif arg == "state0":
            settings["state0"] = int(flag[1])

        elif arg == "num_particles":
            b_settings["num_particles"] = int(flag[1])
        elif arg == "boltzmann_vel":
            b_settings["boltzmann_vel"] = bool(int(flag[1]))
        elif arg == "temp":
            b_settings["temp"] = float(flag[1])
        elif arg == "num_cores":
            b_settings["num_cores"] = int(flag[1])
        elif arg == "seeds":
            arr = []
            for seed in flag[1:]:
                if seed == "end":
                    break
                arr.append(int(seed))
            b_settings["seeds"] = np.array(arr, dtype=int)
        elif arg == "verbose":
            b_settings["verbose"] = bool(int(flag[1]))
        else:
            raise ValueError(f"Unrecognized flag '{arg}'")

    return settings, b_settings


def check_settings(settings):
    m = settings["model"]
    dim = m.dim

    # Check position and velocity vectors
    for key in ["v0", "r0"]:
        val = settings[key]
        if hasattr(val, "__len__"):
            val_dim = len(val)
        else:
            val_dim = 1
            settings[key] = np.array([val], dtype=float)

        if val_dim > dim:
            settings[key] = settings[key][:dim]
        elif val_dim < dim:
            settings[key] = np.concatenate(
                settings[key], np.zeros(dim - val_dim))

    # Check coeff and state0
    num_states = m.num_states
    coeff = settings["coeff"]
    if coeff != None:
        if len(coeff) > num_states:
            settings["coeff"] = coeff[:num_states]
        elif len(coeff) < num_states:
            settings["coeff"] = np.concatenate(
                coeff, np.zeros(num_states - len(coeff)))

    settings["state0"] = min(settings["state0"], num_states-1)
    settings["state0"] = max(settings["state0"], 0)

    # Check decoherence
    deco = settings["deco"]
    if deco != None:
        for key in ["delta_R", "delta_P"]:
            val = deco[key]

            if hasattr(val, "__len__"):
                val_dim = len(val)
            else:
                val_dim = 1
                deco[key] = np.array([val])

            if val_dim > dim:
                deco[key] = deco[key][:dim]
            elif val_dim < dim:
                deco[key] = np.concatenate(
                    deco[key], np.zeros(dim - val_dim))

        settings["deco"] = deco

    return settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="input file for batch job")
    parser.add_argument("outfile", type=str, help="output file for batch job")
    parser.add_argument("outfolder", type=str, help="dir for verbose output")
    args = parser.parse_args()

    try:
        settings, bs = parse_infile(args.infile)
        settings = check_settings(settings)
        bat = batch.New_Batch(settings, fcns.reached_ground)
        num_particles = bs["num_particles"]
        del bs["num_particles"]
        bat.run(num_particles, args.outfile, args.outfolder, **bs)
    except FileNotFoundError:
        print(f"Error, infile '{args.infile}' not found")

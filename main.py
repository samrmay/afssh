import batch as batch
import models as models
import argparse
import re
import numpy as np


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
            settings["v0"] = np.array(v)
        elif arg == "pos" or arg == "position":
            r = []
            for dim in range(1, len(flag) - 1):
                if flag[dim] == "end":
                    break
                r.append(float(flag[dim]))
            settings["r0"] = np.array(r)
        elif arg == "dt_c":
            settings["dt_c"] = int(flag[1])
        elif arg == "max_iter" or arg == "iter":
            settings["max_iter"] = int(flag[1])
        elif arg == "debug":
            settings["debug"] = bool(flag[1])
        elif arg == "langevin":
            lan = {
                "temp": 298.15,
                "damp": 1e-4
            }
            for i in range(len(flag)):
                item = flag[i]
                if item == "temp":
                    lan["temp"] = flag[i+1]
                if item == "damp":
                    lan["damp"] = flag[i+1]

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
                        deco[key] = np.array(arr)
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
            settings["coeff"] = np.array(coeff)
        elif arg == "t0":
            settings["t0"] = flag[1]
        elif arg == "state0":
            settings["state0"] = flag[1]

    return settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="input file for batch job")
    args = parser.parse_args()

    try:
        settings = parse_infile(args.infile)
        print(settings)
    except FileNotFoundError:
        print(f"Error, infile '{args.infile}' not found")

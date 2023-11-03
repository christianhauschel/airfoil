"""Program to morph airfoils with FFD points"""


# %% ============================================================================
# Modules
# ==============================================================================

import argparse
from airfoil import Airfoil
import numpy as np
import proplot as pplt
from pathlib import Path
import yaml
from copy import copy
from time import perf_counter

if __name__ == "__main__":
    # ==============================================================================
    # Get Parameters
    # ==============================================================================

    parser = argparse.ArgumentParser(description="Script arguments")
    parser.add_argument(
        "fname_config", metavar="fname_config", type=str, help="enter the config fname"
    )
    parser.add_argument("--debug", "-d", help="debug mode", action="store_true")
    args = parser.parse_args()

    with open(args.fname_config) as f:
        config = yaml.safe_load(f)

    # ==============================================================================
    # Calculate
    # ==============================================================================

    t_0 = perf_counter()

    dir_out = Path(config["dir_out"])
    if args.debug:
        print("dir_out", dir_out)

    if args.debug:
        print("FFD...")
    af = Airfoil.load_txt(config["fname_airfoil"])
    af_init = copy(af)
    af.ffd(
        n_pts_chordwise=config["n_pts"],
        fname=dir_out / "ffd.xyz",
        fitted=True,
        xmargin=config["xmargin"] if "xmarign" in config else None,
        ymargin_left=config["ymargin_left"] if "ymargin_left" in config else None,
        ymargin_right=config["ymargin_right"] if "ymargin_right" in config else None,
    )
    if config["plots"]:
        af.plot(ffd=True, fname=dir_out / "airfoil_ffd.png", show=config["show_plots"])
    af.write_plot3d(dir_out / "airfoil.xyz")

    print("Init morph...") if args.debug else 1
    af.morph_init(
        dir_out / "ffd.xyz",
        symmetric=config["symmetric"] if "symmetric" in config else False,
        camber_only=config["camber_only"] if "camber_only" in config else False,
        fix_TE=config["fix_TE"] if "fix_TE" in config else False,
    )

    x = np.array(config["x"])
    af.morph_update(x)

    if config["plots"]:
        af.plot_airfoils(
            [af_init, af],
            show=config["show_plots"],
            fname=dir_out / "airfoil_morphed.png",
            legend=False,
        )

    print("Save result") if args.debug else 1
    af.save_txt(dir_out / "airfoil_morphed.dat")

    t_end = perf_counter()
    print(f"Elapsed time: {t_end-t_0}") if args.debug else 1

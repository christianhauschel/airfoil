"""Test airfoil fitting."""

# %% Modules

import matplotlib.pyplot as plt
import numpy as np
from airfoil import *
from scipy.optimize import minimize, Bounds, linprog, differential_evolution
import proplot as pplt

# %% Parameters

n_pts = 241  # n_pts_tot = 2 * n_pts_half + 1
order_refinement = 1
spacing = "cosine"

args = {
    "n": n_pts,
    # "af_0": af_ref,
    "scaler": 10000,
    "type": "wennerstrom",
}

# %% Load Airfoil to Fit

af_ref = Airfoil.load_txt(fname="/mnt/a/Code/db_airfoils/v007/s100/r000/airfoil.dat")
# af.circular_TE()
af_ref.refine(n_pts, order=order_refinement, spacing=spacing)
af_ref.plot()

# %% Initial Value

if args["type"] == "wennerstrom":
    x0 = np.array(
        [
            20.0,
            -10.0,
            0.05,
            0.025,
            0.08,
            0.5,
            2.0,
            1.0,
        ]
    )

    af_0 = Airfoil.wennerstrom(
        n=n_pts,
        beta_in=x0[0],
        beta_out=x0[1],
        t_LE=x0[2],
        t_TE=x0[3],
        t_max=x0[4],
        u_t_max=x0[5],
        elliptical_ratio_LE=x0[6],
        elliptical_ratio_TE=x0[7],
        name="af_0",
    )

elif args["type"] == "naca4":
    x0 = np.array([6, 5, 4])
    str_number = str(int(x0[0])) + str(int(x0[1])) + f"{int(x0[2]):02d}"
    af_0 = Airfoil.naca(str_number)

af_0.plot()

args["af_0"] = af_ref

# %% Optimization Setup


def obj(x: np.array, args: dict):
    if args["type"] == "wennerstrom":
        af = Airfoil.wennerstrom(
            n=args["n"],
            beta_in=x[0],
            beta_out=x[1],
            t_LE=x[2],
            t_TE=x[3],
            t_max=x[4],
            u_t_max=x[5],
            elliptical_ratio_LE=x[6],
            elliptical_ratio_TE=x[7],
            name="af_0",
        )
    elif args["type"] == "naca4":
        str_number = str(int(x[0])) + str(int(x[1])) + f"{int(x[2]):02d}"
        print(str_number)
        af = Airfoil.naca(str_number)

    try:
        # af.refine(args["n"], order=order_refinement, spacing=spacing)
        af_0 = args["af_0"]

        # MSE
        obj_lower = (
            np.sum((af.lower[:, 1] - af_0.lower[:, 1]) ** 2.0)
            / args["n"]
            * args["scaler"]
        )
        obj_upper = (
            np.sum((af.upper[:, 1] - af_0.upper[:, 1]) ** 2.0)
            / args["n"]
            * args["scaler"]
        )
        mse = (obj_lower + obj_upper) / 2.0

    except Exception as e:
        print(e)
        mse = 1.0

    print("Obj:", mse)
    return mse


# %% Minimize

options = {"maxiter": 10000}

if args["type"] == "naca4":
    bounds = Bounds(lb=[0, 0, 0], ub=[9, 9, 99])
else:
    bounds = None

if args["type"] == "naca4":
    res = minimize(
        obj,
        x0=x0,
        args=args,
        bounds=bounds,
    )
else:
    res = minimize(
        obj,
        method="Nelder-Mead",
        # method="SLSQP",
        x0=x0,
        args=args,
        # options=options,
        # constraints=(),
        bounds=bounds,
    )

# %% Results

# x = np.array([ 8.06482084, -6.52096757, -0.02014074,  0.06541075, -0.01488462,
#         1.00262321,  3.24389903,  0.21141855])

x = res.x

if args["type"] == "wennerstrom":
    af = Airfoil.wennerstrom(
        n=args["n"],
        beta_in=x[0],
        beta_out=x[1],
        # fl1 = args["fl1"],
        # fl2 = args["fl2"],
        t_LE=x[2],
        t_TE=x[3],
        t_max=x[4],
        u_t_max=x[5],
        elliptical_ratio_LE=x[6],
        elliptical_ratio_TE=x[7],
        name="af_fitted",
    )

elif args["type"] == "naca4":
    str_number = str(int(x[0])) + str(int(x[1])) + f"{int(x[2]):02d}"
    af = Airfoil.naca(str_number)

af.plot()


fig, ax = pplt.subplots(figsize=(10, 3))
ax.plot(af_ref.data[:, 0], af_ref.data[:, 1], label="ref")
ax.plot(af.data[:, 0], af.data[:, 1], label="fit")
ax.set_xlim(left=-0.02, right=1.02)
ax.set_aspect("equal")
ax.legend()

pplt.show()

# %% RUN ALL

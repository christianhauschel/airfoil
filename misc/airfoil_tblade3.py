"""Script to test the T-Blade3 Airfoil Parameterization.

References
----------
[1] K. Siddappaji, M. G. Turner, and A. Merchant, “General Capability of 
    Parametric 3D Blade Design Tool for Turbomachinery,” in Volume 8: 
    Turbomachinery, Parts A, B, and C, Copenhagen, Denmark, Jun. 2012, 
    pp. 2331–2344. doi: 10.1115/GT2012-69756.
"""

# %% Modules

import numpy as np
import matplotlib.pyplot as plt
from my_settings import *
from scipy.interpolate import interp1d

# %% Parameters

t = 0.04  # max t/c
beta_in = np.radians(20.0)
beta_out = np.radians(-10.0)
fl1 = 0
fl2 = 0
n_pts = 2000
n_pts_final = 241

# Thickness
t_LE = (0.05,)
t_TE = (0.025,)
t_max = (0.08,)
u_t_max = (0.5,)
elliptical_ratio_LE = (2,)
elliptical_ratio_TE = (1,)

# %% Preliminary Calculations

s1 = np.tan(beta_in)
s2 = np.tan(beta_out)
u1 = fl1 * np.cos(beta_in)
c1 = fl1 * np.sin(beta_in)
u2 = 1.0 - fl2 * np.cos(beta_out)
c2 = -fl2 * np.sin(beta_out)


# %% Camber Line
# u = (1.0 - np.cos(np.linspace(np.pi, 0, n_pts))) / 2.0
u = np.linspace(0, 1, n_pts)
xb = u2 - u1
dd = c1
cc = s1
ub = u - u1

aa = s1 + s2 - 2.0 * (c2 - dd) / xb
bb = (-s1 * xb - aa * xb**3.0 + c2 - dd) / xb**2.0

cam_u = 3 * aa * ub**2.0 + 2.0 * bb * ub + cc
cam = aa * ub**3.0 + bb * ub**2.0 + cc * ub + dd

# %% Thickness


def thickness_wennerstrom(
    u_in: float = 0,
    t_LE: float = 0.05,
    t_TE: float = 0.025,
    t_max: float = 0.08,
    u_t_max: float = 0.5,
    elliptical_ratio_LE: float = 2,
    elliptical_ratio_TE: float = 1,
    t_multiplier: float = 1,
) -> float:
    """Computes the Wennerstrom thickness with an elliptical LE/TE at given pt.

    Parameters
    ----------
    u_in : float
        chord-line point at index i = u(i)
    t_LE : float
        LE thickness
    t_TE : float
        TE thickness
    t_max : float
        maximum thickness
    u_t_max : float
        chordwise location of maximum thickness
    elliptical_ratio_LE : float
        elliptic TE ratio
    elliptical_ratio_TE : float
        elliptic LE ratio
    t_multiplier : float
        thickness multiplier value

    Returns
    -------
    float
        thickness
    """
    thick = 0

    # Initialize i_le and i_te and compute u_t_max_ (location of max thickness)
    if u_t_max < 0.5:
        t_1 = t_TE
        t_2 = t_LE
        u_t_max_ = 1.0 - u_t_max
    else:
        t_1 = t_LE
        t_2 = t_TE
        u_t_max_ = u_t_max

    # Compute Wennerstrom coefficients
    aa = -((0.5 * t_max) - (0.5 * t_1)) / (2.0 * (u_t_max_**3))
    bb = 0.0
    cc = 3.0 * ((0.5 * t_max) - (0.5 * t_1)) / (2.0 * u_t_max_)
    dd = 0.5 * t_1
    ee = (
        3.0 * ((0.5 * t_max) - (0.5 * t_1)) / (2.0 * (u_t_max_**2) * (1.0 - u_t_max_))
    )
    ee = ee - ((0.5 * t_max) - (0.5 * t_2)) / (1.0 - u_t_max_) ** 3
    ff = -3.0 * ((0.5 * t_max) - (0.5 * t_1)) / (2.0 * (u_t_max_**2))
    gg = 0.0
    hh = 0.5 * t_max

    if u_t_max < 0.5:
        ub1 = 1.0 - u_t_max_
        y1 = (ee * (ub1**3)) + (ff * (ub1**2)) + (gg * ub1) + hh
        yp1 = -((3.0 * ee * (ub1**2)) + (2.0 * ff * ub1) + gg)
        y2 = dd
        yp2 = cc
    else:
        y1 = dd
        yp1 = cc
        ub1 = 1.0 - u_t_max_
        y2 = (ee * (ub1**3)) + (ff * (ub1**2)) + (gg * ub1) + hh
        yp2 = -((3.0 * ee * (ub1**2)) + (2.0 * ff * ub1) + gg)

    yyp1 = y1 * yp1
    x1 = np.sqrt(
        (yyp1 * (elliptical_ratio_LE**2)) ** 2
        + ((elliptical_ratio_LE**2) * (y1**2))
    ) - (yyp1 * (elliptical_ratio_LE**2))
    ale = (yyp1 * (elliptical_ratio_LE**2)) + x1
    ble = ale / elliptical_ratio_LE

    yyp2 = y2 * yp2
    x2 = np.sqrt(
        (yyp2 * (elliptical_ratio_TE**2)) ** 2
        + ((elliptical_ratio_TE**2) * (y2**2))
    ) - (yyp2 * (elliptical_ratio_TE**2))
    ate = (yyp2 * (elliptical_ratio_TE**2)) + x2
    # bte = ate / elliptical_ratio_TE

    uscale = 1.0 + x1 + x2
    u = -x1 + (uscale * u_in)

    ub = 1.0 - u if (u_t_max < 0.5) else u

    # Compute Wennerstrom thickness
    if u <= 0.0:
        return (
            np.sqrt(((ale**2) - (u + x1 - ale) ** 2) / (elliptical_ratio_LE**2))
            if thick != 0
            else np.sqrt(
                ((ale**2) - (u + x1 - ale) ** 2) / (elliptical_ratio_LE**2)
            )
            * t_multiplier
        )

    elif u >= (1.0 + x2):
        return 0.0

    elif u >= 1.0:
        return (
            np.sqrt(
                ((ate**2) - (u - (1.0 + x2) + ate) ** 2) / (elliptical_ratio_TE**2)
            )
            if thick != 0
            else np.sqrt(
                ((ate**2) - (u - (1.0 + x2) + ate) ** 2) / (elliptical_ratio_TE**2)
            )
            * t_multiplier
        )

    elif ub < u_t_max_:
        return ((aa * (ub**3)) + (bb * (ub**2)) + (cc * ub) + dd) * t_multiplier
    else:
        ub1 = ub - u_t_max_
        return ((ee * (ub1**3)) + (ff * (ub1**2)) + (gg * ub1) + hh) * t_multiplier


# % Test
u = np.linspace(0, 1, n_pts)
t = np.zeros(n_pts)
for i in range(n_pts):
    t[i] = thickness_wennerstrom(u[i])

# %% Suction and Pressure Sides
ang = np.arctan(cam_u)
u_bbot = u + t * np.sin(ang)
v_bbot = cam - t * np.cos(ang)
u_btop = u - t * np.sin(ang)
v_btop = cam + t * np.cos(ang)

# %% Interpolate

u_int_lower = (1.0 - np.cos(np.linspace(np.pi, 0, n_pts_final))) / 2.0
u_int_upper = (np.cos(np.linspace(np.pi, 0, n_pts_final))) / 2.0 + 0.5
lower_int = interp1d(u, v_bbot, kind="cubic")(u_int_lower)
upper_int = interp1d(u, v_btop, kind="cubic")(u_int_upper)

# %% Plots

fig, ax = pplt.subplots(figsize=plt.figaspect(1) * 2)
ax.plot(u, cam, "-", label="Airfoil T-Blade3")
ax.plot(u_int_lower, lower_int, ".-", label="lower")
ax.plot(u_int_upper, upper_int, ".-", label="upper")
# ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
ax.format(title="Wennerstrom Thickness", xlabel="u", ylabel="v")
ax.set_aspect(True)

pplt.show()

af = np.vstack(
    (np.array([u_int_lower, lower_int]).T, np.array([u_int_upper, upper_int]).T)
)

# %% RUN ALL

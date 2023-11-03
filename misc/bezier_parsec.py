"""Tests the Bézier-PARSEC (BP) 3333 parameterization (12 parameters)."""

# %% Modules

import numpy as np
import splipy.curve_factory as cf
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from copy import deepcopy

# %% Parameters

r_LE = -0.009529987
x_t = 0.254124625058989
y_t = 0.0361931947755189
k_t = -0.328482167
beta_TE = 0.39538008583443
gamma_LE = 0.137964237942989
x_c = 0.475647759705384
y_c = 0.0384698521340468
k_c = -0.297039510401247
alpha_TE = 0.152907885489628
dz_TE = 0.0
z_TE = 0.0

# %% Parameter r_t

lb = np.max(np.array([0, x_t - np.sqrt(-2 * y_t / 3 / k_t)]))
ub = x_t


def f(r_t):
    return (
        (27 / 4) * k_t**2 * r_t**4
        - 27 * k_t**2 * x_t * r_t**3
        + (9 * k_t * y_t + (81 / 2) * k_t**2 * x_t**2) * r_t**2
        + (2 * r_LE - 18 * k_t * x_t * y_t - 27 * k_t**2 * x_t**3) * r_t
        + (3 * y_t**2 + 9 * k_t * x_t**2 * y_t + (27 / 4) * k_t**2 * x_t**4)
    )


r_t = fminbound(f, lb, ub)

# %% LE Thickness Curve

pts_t_LE = np.zeros((4, 2))

# x
pts_t_LE[0, 0] = 0.0
pts_t_LE[1, 0] = 0.0
pts_t_LE[2, 0] = r_t
pts_t_LE[3, 0] = x_t

# y
pts_t_LE[0, 1] = 0.0
pts_t_LE[1, 1] = 3 * k_t * (x_t - r_t) ** 2.0 / 2.0 + y_t
pts_t_LE[2, 1] = y_t
pts_t_LE[3, 1] = y_t

crv_t_LE = cf.bezier(pts_t_LE, quadratic=False).reparam()

# %% TE Thickness Curve


def cot(x):
    return np.cos(x) / np.sin(x)


pts_t_TE = np.zeros((4, 2))

# x
pts_t_TE[0, 0] = x_t
pts_t_TE[1, 0] = 2 * x_t - r_t
pts_t_TE[2, 0] = 1 + (dz_TE - (3 * k_t * (x_t - r_t) ** 2.0 / 2.0 + y_t)) * cot(beta_TE)
pts_t_TE[3, 0] = 1

# y
pts_t_TE[0, 1] = y_t
pts_t_TE[1, 1] = y_t
pts_t_TE[2, 1] = 3 * k_t * (x_t - r_t) ** 2.0 / 2 + y_t
pts_t_TE[3, 1] = dz_TE

crv_t_TE = cf.bezier(pts_t_TE, quadratic=False).reparam()

# %% Parameter r_c

lb_1 = 0.0
ub_1 = y_c


def f_1(r_t):
    return (
        16 + 3 * k_c * (cot(gamma_LE) + cot(alpha_TE)) * (1 + z_TE * cot(alpha_TE))
    ) / (3 * k_c * (cot(gamma_LE) + cot(alpha_TE))) + 4.0 * np.sqrt(
        16
        + 6
        * k_c
        * (cot(gamma_LE) + cot(alpha_TE))
        * (1 - y_c * cot(gamma_LE) + cot(alpha_TE))
        + z_TE * cot(alpha_TE)
    )


r_c = fminbound(f_1, lb_1, ub_1)

# %% LE Camber Curve

pts_c_LE = np.zeros((4, 2))

# x
pts_c_LE[0, 0] = 0
pts_c_LE[1, 0] = r_c * cot(gamma_LE)
pts_c_LE[2, 0] = x_c - np.sqrt(2 * (r_c - y_c) / 3 / k_c)
pts_c_LE[3, 0] = x_c

# y
pts_c_LE[0, 1] = 0
pts_c_LE[1, 1] = r_c
pts_c_LE[2, 1] = y_c
pts_c_LE[3, 1] = y_c

crv_c_LE = cf.bezier(pts_c_LE, quadratic=False).reparam()

# %% TE Camber Curve

pts_c_TE = np.zeros((4, 2))

# x
pts_c_TE[0, 0] = x_c
pts_c_TE[1, 0] = x_c + np.sqrt(2 * (r_c - y_c) / 3 / k_c)
pts_c_TE[2, 0] = 1 + (z_TE - r_c) * cot(alpha_TE)
pts_c_TE[3, 0] = 1

# y
pts_c_TE[0, 1] = y_c
pts_c_TE[1, 1] = y_c
pts_c_TE[2, 1] = r_c
pts_c_TE[3, 1] = z_TE

crv_c_TE = cf.bezier(pts_c_TE, quadratic=False).reparam()


# %% Evaluate

t = np.linspace(0, 1, 2000)
t_LE = crv_t_LE.evaluate(t)
t_TE = crv_t_TE.evaluate(t)
c_LE = crv_c_LE.evaluate(t)
c_TE = crv_c_TE.evaluate(t)

t = np.vstack((t_LE, t_TE))
c = np.vstack((c_LE, c_TE))

upper = t
lower = deepcopy(t)
lower[:, 1] = lower[:, 1] - c[:, 1]

# %% Plot

fig, ax = plt.subplots()
# ax.plot(crv_t_LE[:,0], crv_t_LE[:,1], ".")
# ax.plot(crv_t_TE[:,0], crv_t_TE[:,1], ".")
# ax.plot(crv_c_LE[:,0], crv_c_LE[:,1], ".")
# ax.plot(crv_c_TE[:,0], crv_c_TE[:,1], ".")

# ax.plot(t_LE[:,0], t_LE[:,1])
# ax.plot(t_TE[:,0], t_TE[:,1])
# ax.plot(c_LE[:,0], c_LE[:,1])
# ax.plot(c_TE[:,0], c_TE[:,1])

ax.plot(c[:, 0], c[:, 1], "b", label="camber")
ax.plot(upper[:, 0], upper[:, 1], "r", label="upper")
# ax.plot(lower[:,0], lower[:,1], "g", label="lower")

ax.set_aspect(True)

# %% RUN ALL

plt.show()

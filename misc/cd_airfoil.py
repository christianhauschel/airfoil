"""
Testing CD airfoil parameterization.

Source: https://patentimages.storage.googleapis.com/ca/85/ad/af89b9e2d4603a/US4431376.pdf
"""


# %% Modules

from airfoil import Airfoil
from my_settings import pplt, np
from copy import copy

# %% Parameters

beta_l_star = np.radians(20.0)
theta_t_star = np.radians(18.5)
tau = 0.4  # TODO
b_t = 1.0
m_t = 0.04

# %% Camber Line

alpha_ch = beta_l_star + 0.5 * theta_t_star

l = tau * np.sin(np.radians(90.0) - alpha_ch)

L_fcs = l / b_t


def b_ratio(L_fcs):
    if L_fcs > 0.0 and L_fcs <= 0.77:
        return 0.61 - 0.26 * L_fcs
    elif L_fcs > 0.77 and L_fcs < 1.0:
        return 0.4


def theta_ratio(L_fcs):
    if L_fcs > 0.0 and L_fcs <= 0.77:
        return 0.87 - 0.77 * L_fcs
    elif L_fcs > 0.77 and L_fcs < 1.0:
        return 0.27


ratio_b_ft = b_ratio(L_fcs)
ratio_theta_ft = theta_ratio(L_fcs)

b_f = ratio_b_ft * b_t
theta_f_star = ratio_theta_ft * theta_t_star

# -------------------------------------
# Arc 1
# -------------------------------------

# Line: TMCF
gamma_tmcf = alpha_ch - beta_l_star
v_tmcf = np.zeros(2)
v_tmcf[0] = 1.0
v_tmcf[1] = np.tan(gamma_tmcf) * v_tmcf[0]
v_tmcf /= np.linalg.norm(v_tmcf)


def tmcf(t):
    return t * v_tmcf


# Line: TFC
gamma_tfc = gamma_tmcf - theta_f_star
v_tfc = np.zeros(2)
v_tfc[0] = 1.0
v_tfc[1] = np.tan(gamma_tfc) * v_tfc[0]
v_tfc /= np.linalg.norm(v_tfc)


# Line: TMCR
gamma_tmcr = gamma_tmcf - theta_t_star
v_tmcr = np.zeros(2)
v_tmcr[0] = 1.0
v_tmcr[1] = np.tan(gamma_tmcr) * v_tmcr[0]
v_tmcr /= np.linalg.norm(v_tmcr)


def tmcr(t):
    return t * v_tmcr + np.array([b_t, 0.0])


# %% Plots

t_vec = np.linspace(0, 10, 100)

fig, ax = pplt.subplots()
for t in t_vec:
    ax.plot(
        tmcf(t)[0],
        tmcf(t)[1],
        ".",
    )


# %%


def solution(n, k):
    """Solution of cos t - sin t = k

    For k + 1 /= 0.

    n is an integer.
    """

    t1 = 2 * (np.arctan((1 - np.sqrt(2 - k**2.0)) / (k + 1)) + np.pi * n)

    t2 = 2 * (np.arctan((1 + np.sqrt(2 - k**2.0)) / (k + 1)) + np.pi * n)

    return t1, t2


# Constraint 1
k = v_tmcf[0] - v_tmcf[1]
n = 0
t1, t2 = solution(n=0, k=k)

print(np.degrees(t1), np.degrees(t2))
t_fa_xa = copy(t2)


# Constraint 3
k = v_tmcr[0] - v_tmcr[1]
n = 0
t1, t2 = solution(n=0, k=k)

print(np.degrees(t1), np.degrees(t2))
t_ra_xc = copy(t2)


# Constraint 2
k = v_tfc[0] - v_tfc[1]
n = 0
t1, t2 = solution(n=0, k=k)
print(np.degrees(t1), np.degrees(t2))
t_fa_xb = copy(t2)
t_ra_xb = copy(t2)

# Solve linear system

A = np.array(
    [
        [1, 0, 0, 0, np.cos(t_fa_xa), 0],
        [0, 1, 0, 0, np.sin(t_fa_xa), 0],
        [0, 0, 1, 0, 0, np.cos(t_ra_xc)],
        [0, 0, 0, 1, 0, np.sin(t_ra_xc)],
        [1, 0, -1, 0, np.cos(t_fa_xb), -np.cos(t_ra_xb)],
        [0, 1, 0, -1, np.sin(t_fa_xb), -np.sin(t_ra_xb)],
    ]
)

b = np.array([0, 0, b_t, 0, 0, 0])

x = np.linalg.solve(A, b)

x_fa = x[:2]
x_ra = x[2:4]
r_fa = x[4]
r_ra = x[-1]

print("x_fa: ", x_fa, "\nx_ra: ", x_ra)
print("r_fa: ", r_fa, "\nr_ra: ", r_ra)


# %% Thickness


# Ba1
def f_mt_ratio(L_fcs):
    if L_fcs > 0.0 and L_fcs <= 0.77:
        return 0.367 - 0.087 * L_fcs
    elif L_fcs > 0.77 and L_fcs < 1.0:
        return 0.33


mt_ratio = f_mt_ratio(L_fcs)
loc = b_t / m_t * f_mt_ratio(L_fcs)

# Ba2


# %% RUN ALL

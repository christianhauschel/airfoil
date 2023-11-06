"""Functions of different kinds of airfoil parameterizations."""

import numpy as np
from scipy.interpolate import interp1d
from .geometry import *
from .sampling import sampling


def naca4(number: str, n: int, finite_TE=False, spacing="cosine", **kwargs):
    """
    Returns 2*n+1 points in [0 1] for the given 4 digit NACA number string.
    """

    m = float(number[0]) / 100.0
    p = float(number[1]) / 10.0
    t = float(number[2:]) / 100.0
    x = sampling(spacing, 0.0, 1.0, n + 1, **kwargs)

    # Camber
    y_c = np.zeros_like(x)
    if p != 0.0:
        mask = np.logical_and(x >= 0, x <= p)
        y_c[mask] = m / p**2.0 * (2.0 * p * x[mask] - x[mask] ** 2.0)
        y_c[~mask] = (
            m
            / (1.0 - p) ** 2.0
            * ((1.0 - 2.0 * p) + 2.0 * p * x[~mask] - x[~mask] ** 2.0)
        )

    # Thickness
    y_t = (
        5.0
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2.0
            + 0.2843 * x**3.0
            - 0.1015 * x**4.0
        )
    )

    dycdx = np.zeros_like(x)
    if p != 0.0:
        dycdx[mask] = 2.0 * m / p**2.0 * (p - x[mask])
        dycdx[~mask] = 2.0 * m / (1.0 - p) ** 2.0 * (p - x[~mask])
    theta = np.arctan(dycdx)

    x_u = x - y_t * np.sin(theta)
    x_l = x + y_t * np.sin(theta)
    y_u = y_c + y_t * np.cos(theta)
    y_l = y_c - y_t * np.cos(theta)

    # Concatenate the upper and lower surface coordinates and flip the lower
    # surface to get clockwise orientation
    x = np.concatenate((np.flip(x_u), np.flip(x_l[::-1])))
    y = np.concatenate((np.flip(y_u), np.flip(y_l[::-1])))

    return x, y


# def naca5(number: str, n: int, finite_TE=False, spacing="cosine", **kwargs):
#     """
#     Returns 2*n+1 points in [0 1] for the given 5 digit NACA number string.

#     References
#     ----------
#     https://github.com/dgorissen/naca
#     """

#     naca1 = int(number[0])
#     naca23 = int(number[1:3])
#     naca45 = int(number[3:])

#     cld = naca1 * (3.0 / 2.0) / 10.0
#     p = 0.5 * naca23 / 100.0
#     t = naca45 / 100.0

#     a0 = +0.2969
#     a1 = -0.1260
#     a2 = -0.3516
#     a3 = +0.2843

#     a4 = -0.1015 if finite_TE else -0.1036
#     x = sampling(spacing, 0, 1, n + 1, **kwargs)

#     yt = [
#         5
#         * t
#         * (a0 * np.sqrt(xx) + a1 * xx + a2 * xx**2 + a3 * xx**3 + a4 * xx**4)
#         for xx in x
#     ]

#     P = [0.05, 0.1, 0.15, 0.2, 0.25]
#     M = [0.0580, 0.1260, 0.2025, 0.2900, 0.3910]
#     K = [361.4, 51.64, 15.957, 6.643, 3.230]

#     m = interpolate(P, M, [p])[0]
#     k1 = interpolate(M, K, [m])[0]

#     xc1 = [xx for xx in x if xx <= p]
#     xc2 = [xx for xx in x if xx > p]
#     xc = xc1 + xc2

#     if p == 0:
#         xu = x
#         yu = yt

#         xl = x
#         yl = [-x for x in yt]

#         zc = [0] * len(xc)
#     else:
#         yc1 = [
#             k1 / 6.0 * (xx**3 - 3 * m * xx**2 + m**2 * (3 - m) * xx) for xx in xc1
#         ]
#         yc2 = [k1 / 6.0 * m**3 * (1 - xx) for xx in xc2]
#         zc = [cld / 0.3 * xx for xx in yc1 + yc2]

#         dyc1_dx = [
#             cld / 0.3 * (1.0 / 6.0) * k1 * (3 * xx * 2 - 6 * m * xx + m**2 * (3 - m))
#             for xx in xc1
#         ]
#         dyc2_dx = [cld / 0.3 * (1.0 / 6.0) * k1 * m**3] * len(xc2)

#         dyc_dx = dyc1_dx + dyc2_dx
#         theta = [np.arctan(xx) for xx in dyc_dx]

#         xu = [xx - yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)]
#         yu = [xx + yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

#         xl = [xx + yy * np.sin(zz) for xx, yy, zz in zip(x, yt, theta)]
#         yl = [xx - yy * np.cos(zz) for xx, yy, zz in zip(zc, yt, theta)]

#     X = xu[::-1] + xl[1:]
#     Z = yu[::-1] + yl[1:]

#     return X, Z


def naca5(number:str, n: int, finite_TE=False, spacing="cosine", **kwargs):

    xx = np.zeros(n)
    y_t = np.zeros(n)
    y_c = np.zeros(n)
    x = np.zeros(2 * n)
    y = np.zeros(2 * n)

    # TE point bunching parameter
    an = 1.5

    number = int(number)

    n5 = number // 10000
    n4 = (number - n5 * 10000) // 1000
    n3 = (number - n5 * 10000 - n4 * 1000) // 100
    n2 = (number - n5 * 10000 - n4 * 1000 - n3 * 100) // 10
    n1 = number - n5 * 10000 - n4 * 1000 - n3 * 100 - n2 * 10

    n543 = 100 * n5 + 10 * n4 + n3

    if n543 == 210:
        m, c = 0.0580, 361.4
    elif n543 == 220:
        m, c = 0.1260, 51.64
    elif n543 == 230:
        m, c = 0.2025, 15.957
    elif n543 == 240:
        m, c = 0.2900, 6.643
    elif n543 == 250:
        m, c = 0.3910, 3.230
    else:
        print('Illegal 5-digit designation')
        print('First three digits must be 210, 220, ... 250')
        number = 0
        return

    t = (n2 * 10 + n1) / 100.0
    anp = an + 1.0

    for i in range(n):
        frac = i / (n - 1)
        if i == n - 1:
            xx[i] = 1.0
        else:
            xx[i] = 1.0 - anp * frac * (1.0 - frac)**an - (1.0 - frac)**anp

        y_t[i] = (0.29690 * np.sqrt(xx[i]) - 0.12600 * xx[i] - 0.35160 * xx[i]**2 +
                 0.28430 * xx[i]**3 - 0.10150 * xx[i]**4) * t / 0.20

        if xx[i] < m:
            y_c[i] = (c / 6.0) * (xx[i]**3 - 3.0 * m * xx[i]**2 + m**2 * (3.0 - m) * xx[i])
        else:
            y_c[i] = (c / 6.0) * m**3 * (1.0 - xx[i])

    ib = 0
    for i in range(n - 1, -1, -1):
        ib += 1
        x[ib] = xx[i]
        y[ib] = y_c[i] + y_t[i]

    for i in range(1, n):
        ib += 1
        x[ib] = xx[i]
        y[ib] = y_c[i] - y_t[i]


    return x[1:], y[1:]


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
    id_smaller0 = u < 0.0
    id_bigger1x2 = u >= 1.0 + x2
    id_bigger1 = u >= 1.0
    id_ub = ub < u_t_max_
    id_rest = np.logical_not(id_ub)

    ub1 = ub - u_t_max_
    t = np.zeros(len(u))

    t[id_rest] = (
        (ee * (ub1[id_rest] ** 3))
        + (ff * (ub1[id_rest] ** 2))
        + (gg * ub1[id_rest])
        + hh
    ) * t_multiplier

    t[id_ub] = (
        (aa * (ub[id_ub] ** 3)) + (bb * (ub[id_ub] ** 2)) + (cc * ub[id_ub]) + dd
    ) * t_multiplier

    t[id_bigger1] = (
        np.sqrt(
            ((ate**2) - (u[id_bigger1] - (1.0 + x2) + ate) ** 2)
            / (elliptical_ratio_TE**2)
        )
        * t_multiplier
    )

    t[id_bigger1x2] = np.zeros(len(t[id_bigger1x2]))

    t[id_smaller0] = (
        np.sqrt(
            ((ale**2) - (u[id_smaller0] + x1 - ale) ** 2) / (elliptical_ratio_LE**2)
        )
        * t_multiplier
    )

    return t
    # # Compute Wennerstrom thickness
    # if u <= 0.0:
    #     return (
    #         np.sqrt(((ale**2) - (u + x1 - ale) ** 2) / (elliptical_ratio_LE**2))
    #         if thick != 0
    #         else np.sqrt(
    #             ((ale**2) - (u + x1 - ale) ** 2) / (elliptical_ratio_LE**2)
    #         )
    #         * t_multiplier
    #     )

    # elif u >= (1.0 + x2):
    #     return 0.0

    # elif u >= 1.0:
    #     return (
    #         np.sqrt(
    #             ((ate**2) - (u - (1.0 + x2) + ate) ** 2) / (elliptical_ratio_TE**2)
    #         )
    #         if thick != 0
    #         else np.sqrt(
    #             ((ate**2) - (u - (1.0 + x2) + ate) ** 2) / (elliptical_ratio_TE**2)
    #         )
    #         * t_multiplier
    #     )

    # elif ub < u_t_max_:
    #     return ((aa * (ub**3)) + (bb * (ub**2)) + (cc * ub) + dd) * t_multiplier
    # else:
    #     ub1 = ub - u_t_max_
    #     return ((ee * (ub1**3)) + (ff * (ub1**2)) + (gg * ub1) + hh) * t_multiplier


# circular airfoil
def ellipse(a, b, n: int = 120):
    t = np.linspace(0, 2 * np.pi, n)
    x, y = a * np.cos(t), b * np.sin(t)

    return x, y


def tblade3_wennerstrom(
    n_lower: int = 120,
    beta_in=20.0,
    beta_out=-10.0,
    fl1=0.0,
    fl2=0.0,
    t_LE=0.05,
    t_TE=0.025,
    t_max=0.08,
    u_t_max=0.5,
    elliptical_ratio_LE=2,
    elliptical_ratio_TE=1,
):
    """Airfoil parameterization according to the blade modeler `T-Blade3`.

    References
    ----------
    [1] K. Siddappaji, M. G. Turner, and A. Merchant, “General Capability of
        Parametric 3D Blade Design Tool for Turbomachinery,” in Volume 8:
        Turbomachinery, Parts A, B, and C, Copenhagen, Denmark, Jun. 2012,
        pp. 2331–2344. doi: 10.1115/GT2012-69756.
    """

    n_pts = 2000

    beta_in = np.radians(beta_in)
    beta_out = np.radians(beta_out)

    #  Preliminary Calculations
    s1 = np.tan(beta_in)
    s2 = np.tan(beta_out)
    u1 = fl1 * np.cos(beta_in)
    c1 = fl1 * np.sin(beta_in)
    u2 = 1.0 - fl2 * np.cos(beta_out)
    c2 = -fl2 * np.sin(beta_out)

    # Camber Line
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

    # % Test
    u = np.linspace(0, 1, n_pts)
    t = np.zeros(n_pts)
    # for i in range(n_pts):
    #     t[i] = thickness_wennerstrom(
    #         u[i],
    #         t_LE=t_LE,
    #         t_TE=t_TE,
    #         t_max=t_max,
    #         u_t_max=u_t_max,
    #         elliptical_ratio_LE=elliptical_ratio_LE,
    #         elliptical_ratio_TE=elliptical_ratio_TE,
    #         t_multiplier=1.0,
    #     )

    t = thickness_wennerstrom(
        u,
        t_LE=t_LE,
        t_TE=t_TE,
        t_max=t_max,
        u_t_max=u_t_max,
        elliptical_ratio_LE=elliptical_ratio_LE,
        elliptical_ratio_TE=elliptical_ratio_TE,
        t_multiplier=1.0,
    )

    # Suction and Pressure Sides
    ang = np.arctan(cam_u)
    u_bbot = u + t * np.sin(ang)
    v_bbot = cam - t * np.cos(ang)
    u_btop = u - t * np.sin(ang)
    v_btop = cam + t * np.cos(ang)

    # Interpolate
    u_int_upper = (1.0 - np.cos(np.linspace(np.pi, 0, n_lower + 1))) / 2.0
    u_int_lower = (np.cos(np.linspace(np.pi, 0, n_lower + 1))) / 2.0 + 0.5
    u_int_lower = u_int_lower[1:]
    lower_int = interp1d(u, v_bbot, kind="cubic")(u_int_lower)
    upper_int = interp1d(u, v_btop, kind="cubic")(u_int_upper)

    return np.vstack(
        (np.array([u_int_upper, upper_int]).T, np.array([u_int_lower, lower_int]).T)
    )


class ParsecParameters(object):
    """Parameters defining a PARSEC airfoil"""

    def __init__(self):
        self.r_le = 0.0  # Leading edge radius
        self.X_up = 0.0  # Upper crest location X coordinate
        self.Z_up = 0.0  # Upper crest location Z coordinate
        self.Z_XX_up = 0.0  # Upper crest location curvature
        self.X_lo = 0.0  # Lower crest location X coordinate
        self.Z_lo = 0.0  # Lower crest location Z coordinate
        self.Z_XX_lo = 0.0  # Lower crest location curvature
        self.Z_te = 0.0  # Trailing edge Z coordinate
        self.dZ_te = 0.0  # Trailing edge thickness
        self.alpha_te = 0.0  # Trailing edge direction angle
        self.beta_te = 0.0  # Trailing edge wedge angle
        self.P_mix = 1.0  # Blending parameter


class ParsecCoefficients(object):
    """
    This class calculates the equation systems which define the coefficients
    for the polynomials given by the parsec airfoil parameters.
    """

    def __init__(self, parsec_params):
        self._a_up = self._calc_a_up(parsec_params)
        self._a_lo = self._calc_a_lo(parsec_params)

    def a_up(self):
        """Returns coefficient vector for upper surface"""
        return self._a_up

    def a_lo(self):
        """Returns coefficient vector for lower surface"""
        return self._a_lo

    def _calc_a_up(self, parsec_params):
        Amat = self._prepare_linsys_Amat(parsec_params.X_up)
        Bvec = np.array(
            [
                parsec_params.Z_te + parsec_params.dZ_te / 2,
                parsec_params.Z_up,
                np.tan(parsec_params.alpha_te - parsec_params.beta_te / 2),
                0.0,
                parsec_params.Z_XX_up,
                np.sqrt(2 * parsec_params.r_le),
            ]
        )
        return np.linalg.solve(Amat, Bvec)

    def _calc_a_lo(self, parsec_params):
        Amat = self._prepare_linsys_Amat(parsec_params.X_lo)
        Bvec = np.array(
            [
                parsec_params.Z_te - parsec_params.dZ_te / 2,
                parsec_params.Z_lo,
                np.tan(parsec_params.alpha_te + parsec_params.beta_te / 2),
                0.0,
                parsec_params.Z_XX_lo,
                -np.sqrt(2 * parsec_params.r_le),
            ]
        )
        return np.linalg.solve(Amat, Bvec)

    def _prepare_linsys_Amat(self, X):
        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [X**0.5, X**1.5, X**2.5, X**3.5, X**4.5, X**5.5],
                [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                [
                    0.5 * X**-0.5,
                    1.5 * X**0.5,
                    2.5 * X**1.5,
                    3.5 * X**2.5,
                    4.5 * X**3.5,
                    5.5 * X**4.5,
                ],
                [
                    -0.25 * X**-1.5,
                    0.75 * X**-0.5,
                    3.75 * X**0.5,
                    8.75 * X**1.5,
                    15.75 * X**2.5,
                    24.75 * X**3.5,
                ],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )


class ParsecAirfoil(object):
    """Airfoil defined by PARSEC Parameters"""

    def __init__(self, parsec_params):
        self._coeff = ParsecCoefficients(parsec_params)

    def Z_up(self, X):
        """Returns Z(X) on upper surface, calculates PARSEC polynomial"""
        a = self._coeff.a_up()
        # print (a)
        return (
            a[0] * X**0.5
            + a[1] * X**1.5
            + a[2] * X**2.5
            + a[3] * X**3.5
            + a[4] * X**4.5
            + a[5] * X**5.5
        )

    def Z_lo(self, X):
        """Returns Z(X) on lower surface, calculates PARSEC polynomial"""
        a = self._coeff.a_lo()
        # print (a)
        return (
            a[0] * X**0.5
            + a[1] * X**1.5
            + a[2] * X**2.5
            + a[3] * X**3.5
            + a[4] * X**4.5
            + a[5] * X**5.5
        )

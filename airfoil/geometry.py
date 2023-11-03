"""Different geometry helper functions."""

import numpy as np
from typing import Optional
from cst import cst, fit
from .sampling import *


def interpolate(xa, ya, queryPoints):
    """Interpolates 2D points using cubic splines.

    A cubic spline interpolation on a given set of points (x,y).
    Recalculates everything on every call which is far from efficient but does
    the job for now should eventually be replaced by an external helper class.
    """

    # PreCompute() from Paint Mono which in turn adapted:
    # NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
    # ISBN 0-521-43108-5, page 113, section 3.3.
    # http://paint-mono.googlecode.com/svn/trunk/src/PdnLib/SplineInterpolator.cs

    # number of points
    n = len(xa)
    u, y2 = [0] * n, [0] * n

    for i in range(1, n - 1):
        # This is the decomposition loop of the tridiagonal algorithm.
        # y2 and u are used for temporary storage of the decomposed factors.

        wx = xa[i + 1] - xa[i - 1]
        sig = (xa[i] - xa[i - 1]) / wx
        p = sig * y2[i - 1] + 2.0

        y2[i] = (sig - 1.0) / p

        ddydx = (ya[i + 1] - ya[i]) / (xa[i + 1] - xa[i]) - (ya[i] - ya[i - 1]) / (
            xa[i] - xa[i - 1]
        )

        u[i] = (6.0 * ddydx / wx - sig * u[i - 1]) / p

    y2[n - 1] = 0

    # This is the backsubstitution loop of the tridiagonal algorithm
    # ((int i = n - 2; i >= 0; --i):
    for i in range(n - 2, -1, -1):
        y2[i] = y2[i] * y2[i + 1] + u[i]

    # interpolate() adapted from Paint Mono which in turn adapted:
    # NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
    # ISBN 0-521-43108-5, page 113, section 3.3.
    # http://paint-mono.googlecode.com/svn/trunk/src/PdnLib/SplineInterpolator.cs

    results = [0] * n

    # loop over all query points
    for i in range(len(queryPoints)):
        # bisection. This is optimal if sequential calls to this
        # routine are at random values of x. If sequential calls
        # are in order, and closely spaced, one would do better
        # to store previous values of klo and khi and test if

        klo = 0
        khi = n - 1

        while khi - klo > 1:
            k = (khi + klo) >> 1
            if xa[k] > queryPoints[i]:
                khi = k
            else:
                klo = k

        h = xa[khi] - xa[klo]
        a = (xa[khi] - queryPoints[i]) / h
        b = (queryPoints[i] - xa[klo]) / h

        # Cubic spline polynomial is now evaluated.
        results[i] = (
            a * ya[klo]
            + b * ya[khi]
            + ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6.0
        )

    return results


def pathlength(pts: np.array) -> float:
    distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.sum(distances)


def distance_2pts(p1, p2):
    """Calculates the distance between two points."""
    return np.linalg.norm(p1 - p2)


def pol2cart(rho, theta, z):
    """Polar to cartesian coordinate transformation."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y, z


def coords2cst(x, y_u, y_l, n_ca, n_th):
    """
    Convert airfoil upper/lower curve coordinates to camber line/thickness distribution CST coefficients.

    Parameters
    ----------
    x : array_like
        X-Coordinates
    y_u, y_l : array_like
        Y-Coordinates of the upper and lower curves, respectively
    n_ca, n_th : int
        Number of CST coefficients to use for the camber line and thickness distribution of the airfoil

    Returns
    -------
    a_ca, a_th : np.ndarray
        CST coefficients describing the camber line and thickness distribution of the airfoil
    t_te : float
        Airfoil trailing edge thickness
    """
    y_c = (y_u + y_l) / 2
    t = y_u - y_l

    a_ca, _ = fit(x, y_c, n_ca, delta=(0.0, 0.0), n1=1)
    a_th, t_te = fit(x, t, n_th)

    return a_ca, a_th, t_te[1]


def cst2coords(a_ca, a_th, t_te, n_coords=100):
    """
    Convert airfoil camber line/thickness distribution CST coefficients to upper/lower curve coordinates.

    Parameters
    ----------
    a_ca, a_th : array_like
        CST coefficients describing the camber line and thickness distribution of the airfoil
    t_te : float
        Airfoil trailing edge thickness
    n_coords : int, optional
        Number of x-coordinates to use. 100 by default

    Returns
    -------
    x : np.ndarray
        Airfoil x-coordinates
    y_u, y_l : np.ndarray
        Airfoil upper and lower curves y-coordinates
    y_c, t : np.ndarray
        Airfoil camber line and thickness distribution
    """
    x = cosine(0, 1, n_coords)
    y_c = cst(x, a_ca, n1=1)
    t = cst(x, a_th, delta=(0, t_te))

    y_u = y_c + t / 2
    y_l = y_c - t / 2
    return x, y_u, y_l, y_c, t

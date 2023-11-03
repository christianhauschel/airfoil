try:
    import proplot as pplt
except:
    import matplotlib.pyplot as pplt

import contextlib
import numpy as np
from scipy.interpolate import interp1d
from .airfoil import Airfoil


def plot_airfoils(
    airfoils: list, fname=None, dpi=300, show=False, legend=True, **kwargs
):
    """Plots a list of airfoils into one figure.

    Parameters
    ----------
    airfoils : list
        list of airfoils to plot
    fname : str, optional
        Filename of image to save, by default None
    dpi : int, optional
        DPI of png file, by default 300
    plot_show : bool, optional
    """
    fig, ax = pplt.subplots(**kwargs)

    y_max = -np.inf
    y_min = np.inf
    chord_max = 0.0
    for af in airfoils:
        ax.plot(af.x, af.y, label=af.name)

        if af.y.max() > y_max:
            y_max = af.y.max()
        if af.y.min() < y_min:
            y_min = af.y.min()
        if af.chord > chord_max:
            chord_max = af.chord

    # Figure Sizing
    fct_size = 1.0 / np.abs(y_max - y_min)
    width = 18 / 2.54
    space = 1.0 if fct_size > 5 else 0.5
    height = width / fct_size + min(1, space)
    fig.set_size_inches(width, height)

    ax.format(
        title="Airfoil Comparison",
        aspect="equal",
        xlim=[-0.01 * chord_max, 1.01 * chord_max],
        xlabel=r"$x$",
        ylabel=r"$y$",
    )
    with contextlib.suppress(Exception):
        if legend:
            ax.legend()

    if show == True:
        pplt.show()
    if fname is not None:
        fig.savefig(fname, dpi=dpi)


def interpolate_airfoils(
    airfoils: list, plot_debug=False, kind=None
) -> list:
    """Interpolates between a list of airfoils.

    Parameters
    ----------
    airfoils : list

    plot_debug : bool, optional
        Show a plot of the interpolated airfoils, by default False

    kind : str, optional
        Kind of interpolation None (automatic selection), "linear", "quadratic", 
        or "cubic". By default None.

    Returns
    -------
    list
    """

    names = [af.name for af in airfoils]

    # plot
    # for af in airfoils:
    #     af.plot()
    n = len(airfoils)
    s = np.linspace(0, 1, n)

    # get max n from airfoils
    n_max = max(af.n for af in airfoils)

    # refine all to n_max
    for af in airfoils:
        af.refine(n_max)

    # generate tensor (n x n_max x 2)
    data = np.array([af.data for af in airfoils])

    n_int = 15
    s_int = np.linspace(0, 1, n_int)

    # interpolate data along dimension 0

    data_int = np.zeros((n_int, n_max, 2))

    for i in range(n_max):
        # data_int[:,i,0] = np.interp(s_int, s,data[:, i, 0])
        # data_int[:,i,1] = np.interp(s_int, s,data[:, i, 1])

        # use interp1d
        if kind is not None:
            kind = kind
        elif n == 3:
            kind = "quadratic"
        elif n > 3:
            kind = "cubic"
        else:
            kind = "linear"

        f0 = interp1d(s, data[:, i, 0], kind=kind)
        f1 = interp1d(s, data[:, i, 1], kind=kind)
        data_int[:, i, 0] = f0(s_int)
        data_int[:, i, 1] = f1(s_int)

    if plot_debug:
        # 3d plot
        import matplotlib.pyplot as plt

        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(projection="3d")

        for i in range(n):
            ax.plot(data[i, :, 0], data[i, :, 1], s[i] * np.ones(n_max), label=names[i])

        for i in range(n_int):
            ax.plot(
                data_int[i, :, 0],
                data_int[i, :, 1],
                s_int[i] * np.ones(n_max),
                c="k",
                alpha=0.5,
                lw=1,
            )

        # grid off
        ax.grid(False)

        plt.show()

        return [Airfoil(data_int[i]) for i in range(n_int)]

"""Main airfoil class.
"""


import contextlib
import itertools
import numpy as np

try:
    import proplot as plt
except:
    import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d
from scipy.interpolate import splrep, splev
from scipy.spatial import distance
from .airfoiltypes import *
from .geometry import *
from polygon_math import polygon
from pathlib import Path
from pickle import load, dump
from copy import copy
from .sampling import sampling
from .math import *
import sys
from splines import Spline


class Airfoil:
    """Airfoil geometry class.

    Features
    --------
    - Imports file and normalizes the data.
    - Calculates various airfoil parameters.
    - Allows modification of the airfoil.
    """

    # ==========================================================================
    # Constructors
    # ==========================================================================

    def __init__(
        self,
        data: np.array,
        name: str = "Noname",
        order=1,
        spacing="cosine",
        n_correction=10,
        **kwargs,
    ):
        """Constructs the airfoil object.

        If the airfoil is not in canonical form, the necessary modifications are made.

        Parameters
        ----------
        data : np.array
            Airfoil data matrix with the columns denoting the x- and y-coordinates.
        name : str, optional
            Name of the airfoil.
        """
        self.name = name
        self.data = data

        # Normalize
        _, self.twist = self._unitize(
            order=order, spacing=spacing, n_correction=n_correction, **kwargs
        )
        self.order = order
        self.spacing = spacing
        self.kwargs = kwargs

        self._recompute()

    def __repr__(self):
        """String represenation of airfoil object."""
        return (
            "────────────────────────────────────────────────────────────────────────────────\n"
            + f" Airfoil           {self.name}\n"
            + "────────────────────────────────────────────────────────────────────────────────\n"
            + f" n                {self.n}\n"
            + f" chord            {self.chord:0.2e}\n"
            + f" thickness_max    {self.thickness_max:0.2e}\n"
            + f" area             {self.area:0.2e}\n"
            + f" I_xx             {self.second_area_moment[0]:0.2e}\n"
            + f" I_yy             {self.second_area_moment[1]:0.2e}\n"
            + f" I_xy             {self.second_area_moment[2]:0.2e}\n"
            + "────────────────────────────────────────────────────────────────────────────────\n"
        )

    def _recompute(self):
        """Computes the airfoil splines."""
        self.s_airfoil = Spline(self.data)
        self.s_camberline = Spline(self.camberline)
        self.s_chord = Spline(self.chordline)
        self.s_upper = Spline(self.upper)
        self.s_lower = Spline(self.lower)

    def normalize(self):
        """Normalizes the airfoil (chord = 1)."""
        self.data /= self._chord
        self._chord = 1.0
        self._recompute()

    def copy(self):
        """Returns a copy of the airfoil."""
        return copy(self)

    @classmethod
    def load_obj(cls, fname: str):
        """Load airfoil object from disc.

        Parameters
        ----------
        fname : str
            fname
        """
        with open(fname, "rb") as f:
            obj = load(f)
        return obj

    @classmethod
    def load_txt(
        cls,
        fname: str,
        name: str = None,
        skiprows: int = 1,
        order: int = 1,
        spacing: str = "cosine",
        **kwargs,
    ):
        """
        Loads an airfoil from a data file.

        Parameters
        ----------
        fname: str
        name: str, optional
        skiprows: int, optional

        Conventions
        -----------
        - The airfoil begins at TE and moves along the upper surface
          counter-clockwise to the pressure side until it reaches again the TE.
        """
        data = np.loadtxt(fname, skiprows=skiprows)

        if name is None:
            if skiprows > 0:
                with open(fname) as f:
                    name = str(f.readline().strip().replace("# ", "").replace("#", ""))
            else:
                name = str(Path(fname).name)
        return cls(data, name=name, order=order, spacing=spacing, **kwargs)

    @classmethod
    def wennerstrom(
        cls,
        chord=1.0,
        n=241,
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
        name="T-Blade3 Wennerstrom",
    ):
        """Creates an airfoil using a T-Blade3-style camber line and a
        Wennerstrom thickness distribution.

        Parameters
        ----------
        n_lower : int, optional
            number of pts of pressure side, by default 241
        beta_in : float, optional
            angle in, by default 20.0
        beta_out : float, optional
            angle out, by default -10.0
        fl1 : float, optional
            ???, by default 0.0
        fl2 : float, optional
            ???, by default 0.0
        t_LE : float, optional
            thickness LE, by default 0.05
        t_TE : float, optional
            thickness TE, by default 0.025
        t_max : float, optional
            thickness max, by default 0.08
        u_t_max : float, optional
            position of max. thickness, by default 0.5
        elliptical_ratio_LE : int, optional
            ratio of LE ellipse, by default 2
        elliptical_ratio_TE : int, optional
            ratio of TE ellipse, by default 1
        name : str, optional
            name, by default "T-Blade3 Wennerstrom"
        """
        assert n % 2, "n must be odd!"

        n_lower = int((n - 1) / 2)

        data = tblade3_wennerstrom(
            n_lower,
            beta_in,
            beta_out,
            fl1,
            fl2,
            t_LE,
            t_TE,
            t_max,
            u_t_max,
            elliptical_ratio_LE,
            elliptical_ratio_TE,
        )

        return cls(data * chord, name=name)

    @classmethod
    def parsec(
        cls,
        n=241,
        r_le=0.01,
        X_up=0.5,
        Z_up=0.03,
        Z_XX_up=-0.1,
        X_lo=0.5,
        Z_lo=-0.03,
        Z_XX_lo=0.1,
        Z_te=0.0,
        dZ_te=0.01,
        alpha_te=-20.0,
        beta_te=20.0,
        name="PARSEC",
        spacing="cosine",
        **kwargs,
    ) -> None:
        assert n % 2, "n must be odd!"

        # Parameters
        params = ParsecParameters()
        params.r_le = r_le
        params.X_up = X_up
        params.Z_up = Z_up
        params.Z_XX_up = Z_XX_up
        params.X_lo = X_lo
        params.Z_lo = Z_lo
        params.Z_XX_lo = Z_XX_lo
        params.Z_te = Z_te
        params.dZ_te = dZ_te
        params.alpha_te = np.radians(alpha_te)
        params.beta_te = np.radians(beta_te)

        # Airfoil
        airfoil = ParsecAirfoil(params)

        # Points
        n_upper = int((n + 1) / 2)
        x = sampling(spacing, 0, 1, n_upper, **kwargs)
        y_lower = airfoil.Z_lo(x)
        y_upper = airfoil.Z_up(x)

        x_final = np.concatenate((np.flip(x), x[1:]))
        y_final = np.concatenate((np.flip(y_upper), y_lower[1:]))

        return cls(np.c_[x_final, y_final], name=name)

    @classmethod
    def naca(
        cls, number: str, chord=1, n=241, finite_TE=False, spacing="cosine", **kwargs
    ):
        """Creates a NACA airfoil.

        Parameters
        ----------
        number : str
            NACA number
        chord : int, optional
            chordlength, by default 1
        n : int, optional
            number of points, must be odd, by default 241
        finite_TE : bool, optional
            _description_, by default False
        spacing : str, optional
            sampling type, by default "cosine"
        """
        assert n % 2, "n must be odd!"

        n_lower = int((n - 1) / 2)

        if len(number) == 4:
            x, y = naca4(number, n_lower, finite_TE, spacing, **kwargs)
        else:
            x, y = naca5(number, n_lower, finite_TE, spacing, **kwargs)

        return cls(np.c_[x * chord, y * chord], name=f"NACA {number}")

    @classmethod
    def ellipse(cls, a=0.5, b=0.5, n=241, name=None, spacing="cosine", **kwargs):
        """Creates a ellipsoidal airfoil.

        Parameters
        ----------
        a : float
            major axis
        b : float
            minor axis
        n : int, optional
            number of points, must be odd, by default 241
        spacing : str, optional
            sampling type: ["cosine", "linear", "conical", "polynomial"], by default "cosine"
        """
        assert n % 2, "n must be odd!"

        x, y = ellipse(a, b, n * 3)

        if name is None:
            name = "circle" if a == b else "ellipse"

        af = cls(np.c_[x, y], name=name)
        af.refine(n, spacing=spacing, **kwargs)

        return af

    @classmethod
    def cst(cls, a_ca, a_th, t_te, chord=1, n=241, name="CST"):
        """
        Creates a CST airfoil.
        """

        assert n % 2, "n must be odd!"

        n_lower = int((n - 1) / 2)
        n_upper = n_lower + 1

        x, y_u, y_l, _, _ = cst2coords(a_ca, a_th, t_te, n_coords=n_upper)

        x2 = np.concatenate((np.flip(x), x[:-1]))
        y2 = np.concatenate((np.flip(y_u), y_l[1:]))
        data2 = np.array((x2, y2)).T

        return cls(data2 * chord, name=name)

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def n(self):
        return len(self.data)

    @property
    def n_lower(self):
        return int((self.n - 1) / 2)

    @property
    def n_upper(self):
        return self.n_lower + 1

    @property
    def centroid(self) -> np.array:
        """Returns the centroid of the airfoil.

        Returns
        -------
        centroid : np.array
            [x, y]
        """
        return polygon(self.data).CenterMass

    @property
    def chord(self) -> float:
        """Returns chord of the airfoil."""
        return self._calculate_chord()

    def _calculate_chord(self):
        return np.linalg.norm(self.LE - self.TE)

    @chord.setter
    def chord(self, chord) -> float:
        """Sets the chord of the airfoil."""
        self.normalize()
        self.data *= chord
        self._chord = chord
        self._recompute()

    @property
    def area(self) -> float:
        """Returns the area of the airfoil."""
        return polygon(self.data).Area

    @property
    def second_area_moment(self) -> tuple:
        """Returns the second area moments of the airfoil.

        Returns
        -------
        second area moments : tuple
            I_xx, I_yy, I_xy
        """
        return tuple(polygon(self.data).SecondMomentArea)

    @property
    def camberline(self):
        """
        Returns the camberline of the airfoil.
        """
        with contextlib.suppress(Exception):
            if np.array_equal(self.upper[:, 0], self.lower[:, 0]):
                return 0.5 * (self.upper + self.lower)
        x = self.upper[:, 0]
        f = interp1d(
            self.lower[:, 0], self.lower[:, 1], kind="linear", fill_value="extrapolate"
        )
        y = f(x).T
        return 0.5 * (self.upper + np.c_[x, y])

    @property
    def chordline(self):
        """
        Returns the chordline of the airfoil.
        """
        return np.array([self.upper[:, 0], np.zeros(len(self.upper[:, 0]))]).T

    @property
    def thickness(self):
        """Starting from the TE.

        Returns
        -------
        thickness : np.array
        """
        return np.abs(self.upper[:, 1] - np.flip(self.lower[:, 1], axis=0))

    @property
    def thickness_max(self):
        """Returns the max. thickness of the airfoil."""
        return np.max(self.thickness)

    @property
    def thickness_te(self):
        """Returns the TE thickness of the airfoil."""
        return self.thickness[0]

    @property
    def lower(self):
        """Returns the pressure side of the airfoil."""
        _, lower = self._upper_lower_split(self.data)
        return lower

    @property
    def upper(self):
        """Returns the suction side of the airfoil."""
        upper, _ = self._upper_lower_split(self.data)
        return upper

    @property
    def x(self):
        """Returns the x-coordinates of the airfoil."""
        return self.data[:, 0]

    @property
    def y(self):
        """Returns the y-coordinates of the airfoil."""
        return self.data[:, 1]

    @property
    def TE(self):
        """Returns the TE coordinates."""
        return self._find_TE(self.data)

    @staticmethod
    def _find_TE(data):
        """Calculates TE from airfoil data.

        Parameters
        ----------
        data : np.array
            Airfoil coordinates.

        Returns
        -------
        np.array
            TE coordinates
        """

        x = (data[0, 0] + data[-1, 0]) / 2
        y = (data[0, 1] + data[-1, 1]) / 2
        return np.array([x, y])

    @staticmethod
    def _find_LE(data, TE):
        dist = distance.cdist(data, np.array([TE]))
        i_TE = np.argmax(dist)
        return data[i_TE], i_TE

    @property
    def LE(self):
        """Returns LE coordinates."""
        return self._find_LE(self.data, self.TE)[0]

    @property
    def id_LE(self):
        """Returns LE index."""

        return self._find_LE(self.data, self.TE)[1]

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def write_plot3d(self, fname):
        """
        This function writes out a 2D airfoil surface in 3D (one element in z direction)
        Parameters
        ----------
        fname : str
                fname to write out, not including the '.xyz' ending
        x : Ndarray [N]
                a list of all the x values of the coordinates
        y : Ndarray [N]
                a list of all the y values of the coordinates
        """

        x = self.x
        y = self.y

        with open(fname, "w") as f:
            f.write("1\n")
            f.write("%d %d %d\n" % (len(x), 2, 1))
            for iDim in range(3):
                for j in range(2):
                    for i in range(len(x)):
                        if iDim == 0:
                            f.write("%g\n" % x[i])
                        elif iDim == 1:
                            f.write("%g\n" % y[i])
                        else:
                            f.write("%g\n" % (float(j)))

    @staticmethod
    def _getClosestY(coords, x):
        """
        Gets the closest y value on the upper and lower point to an x value
        Parameters
        ----------
        coords : Ndarray [N,2]
            coordinates defining the airfoil
        x : float
            The x value to find the closest point for
        Returns
        -------
        yu : float
            The y value of the closest coordinate on the upper surface
        yl : float
            The y value of the closest coordinate on the lower surface
        """

        top = coords[: len(coords + 1) // 2 + 1, :]
        bottom = coords[len(coords + 1) // 2 :, :]

        x_top = np.ones(len(top))
        for i in range(len(top)):
            x_top[i] = abs(top[i, 0] - x)
        yu = top[np.argmin(x_top), 1]
        x_bottom = np.ones(len(bottom))
        for i in range(len(bottom)):
            x_bottom[i] = abs(bottom[i, 0] - x)
        yl = bottom[np.argmin(x_bottom), 1]

        return yu, yl

    @staticmethod
    def _writeFFD(FFDbox, fname):
        """
        This function writes out an FFD in plot3D format from an FFDbox.
        Parameters
        ----------
        FFDBox : Ndarray [N,2,2,3]
            FFD Box to write out
        fname : str
            fname to write out, not including the '.xyz' ending
        """

        n_pts_chordwise = FFDbox.shape[0]

        # Write to file
        with open(f"{fname}", "w") as f:
            f.write("1\n")
            f.write(str(n_pts_chordwise) + " 2 2\n")
            for ell, k, j in itertools.product(range(3), range(2), range(2)):
                for i in range(n_pts_chordwise):
                    f.write("%.15f " % (FFDbox[i, j, k, ell]))
                f.write("\n")

    def _buildFFD(
        self,
        n_pts_chordwise: int,
        fitted: bool,
        xmargin: float,
        ymargin_left: float,
        ymargin_right: float,
        xslice: float,
        coords: np.array,
    ):
        """The function that actually builds the FFD Box from all of the given parameters

        Parameters
        ----------
        n_pts_chordwise : int
            number of FFD points along the chord
        fitted : bool
            flag to pick between a fitted FFD (True) and box FFD (False)
        xmargin : float
            The closest distance of the FFD box to the tip and aft of the airfoil
        ymargin_left : float
            When a box ffd is generated this specifies the top of the box's y values as
            the maximum y value in the airfoil coordinates plus this margin.
            When a fitted ffd is generated this is the margin between the FFD point at
            an xslice location and the upper surface of the airfoil at this location
        ymargin_right : float
            When a box ffd is generated this specifies the bottom of the box's y values as
            the minimum y value in the airfoil coordinates minus this margin.
            When a fitted ffd is generated this is the margin between the FFD point at
            an xslice location and the lower surface of the airfoil at this location
        xslice : Ndarray [N,2]
            User specified xslice locations. If this is chosen n_pts_chordwise is ignored
        coords : Ndarray [N,2]
            the coordinates to use for defining the airfoil, if the user does not
            want the original coordinates for the airfoil used. This shouldn't be
            used unless the user wants fine tuned control over the FFD creation,
            It should be sufficient to ignore.
        """

        if coords is None:
            coords = self.data

        if xslice is None:
            xslice = np.zeros(n_pts_chordwise)
            for i in range(n_pts_chordwise):
                xtemp = i * self._chord / (n_pts_chordwise - self._chord)
                xslice[i] = (
                    min(coords[:, 0])
                    - self._chord * xmargin
                    + (max(coords[:, 0]) + 2.0 * xmargin) * xtemp
                )
        else:
            n_pts_chordwise = len(xslice)

        FFDbox = np.zeros((n_pts_chordwise, 2, 2, 3))

        if fitted:
            ylower = np.zeros(n_pts_chordwise)
            yupper = np.zeros(n_pts_chordwise)
            for i in range(n_pts_chordwise):
                ymargin = ymargin_left + (ymargin_right - ymargin_left) * xslice[i]
                yu, yl = self._getClosestY(coords, xslice[i])
                yupper[i] = yu + ymargin
                ylower[i] = yl - ymargin
        else:
            yupper = np.ones(n_pts_chordwise) * (max(coords[:, 1]) + ymargin_left)
            ylower = np.ones(n_pts_chordwise) * (min(coords[:, 1]) - ymargin_right)

        # X (chordwise)
        FFDbox[:, 0, 0, 0] = xslice[:].copy()
        FFDbox[:, 1, 0, 0] = xslice[:].copy()

        # Y (thicknesswise)

        # Wing Hub
        # lower
        FFDbox[:, 0, 0, 1] = ylower[:].copy()
        # upper
        FFDbox[:, 1, 0, 1] = yupper[:].copy()

        # Wing Tip Copy
        FFDbox[:, :, 1, :] = FFDbox[:, :, 0, :].copy()

        # Z (spanwise)
        FFDbox[:, :, 0, 2] = 0.0
        FFDbox[:, :, 1, 2] = 1.0

        self.pts_ffd_2d = FFDbox[:, :, 0, 0:2]

        return FFDbox

    def ffd(
        self,
        n_pts_chordwise: int,
        fname: str,
        fitted: bool = True,
        xmargin: float = 0.001,
        ymargin_left: float = 0.02,
        ymargin_right: float = 0.02,
        xslice: float = None,
        coords: np.array = None,
    ):
        """
        Generates an FFD from the airfoil and writes it out to file

        Parameters
        ----------
        n_pts_chordwise : int
            the number of chordwise points in the FFD
        fname : str
            fname to write out, not including the '.xyz' ending
        fitted : bool
            flag to pick between a fitted FFD (True) and box FFD (False)
        xmargin : float
            The closest distance of the FFD box to the tip and aft of the airfoil
        ymargin_left : float
            When a box ffd is generated this specifies the top of the box's y values as
            the maximum y value in the airfoil coordinates plus this margin.
            When a fitted ffd is generated this is the margin between the FFD point at
            an xslice location and the upper surface of the airfoil at this location
        ymargin_right : float
            When a box ffd is generated this specifies the bottom of the box's y values as
            the minimum y value in the airfoil coordinates minus this margin.
            When a fitted ffd is generated this is the margin between the FFD point at
            an xslice location and the lower surface of the airfoil at this location
        xslice : Ndarray [N,2]
            User specified xslice locations. If this is chosen n_pts_chordwise is ignored
        coords : Ndarray [N,2]
            the coordinates to use for defining the airfoil, if the user does not
            want the original coordinates for the airfoil used. This shouldn't be
            used unless the user wants fine tuned control over the FFD creation,
            It should be sufficient to ignore.
        """

        try:
            import pygeo
        except:
            raise Exception("Please install pygeo!")

        if xmargin is None:
            xmargin = 0.001
        if ymargin_left is None:
            ymargin_left = 0.02
        if ymargin_right is None:
            ymargin_right = 0.02

        FFDbox = self._buildFFD(
            n_pts_chordwise,
            fitted,
            xmargin,
            ymargin_left,
            ymargin_right,
            xslice,
            coords,
        )

        self.morph_nffd = n_pts_chordwise
        # self.FFDbox = FFDbox
        self._writeFFD(FFDbox, fname)

    def morph_init(
        self,
        fname_ffd,
        axis="y",
        scale=1.0,
        n_pts_ffd=None,
        symmetric=False,
        camber_only=False,
        fix_TE=False,
    ):
        if n_pts_ffd is None:
            try:
                n_pts_ffd = self.morph_nffd
            except:
                print("Please specify n_pts_ffd!")

        self.morph_symmetric = symmetric
        self.morph_camber_only = camber_only
        self.morph_fix_TE = fix_TE

        # Init DVGeo Object
        self.geo = pygeo.DVGeometry(fname_ffd)

        # Create DVgeometry
        af = np.ones((self.n, 3)) * 0.5
        af[:, 0:2] = self.data

        self.geo.addPointSet(af, "airfoil")
        # self.geo.writePointSet("airfoil", "pointset") # write file of current airfoil
        self.geo.addLocalDV(
            "shape",
            lower=None,
            upper=None,
            axis=axis,
            scale=scale,
        )

        # Inputs
        n_x = n_pts_ffd if self.morph_symmetric else n_pts_ffd * 2 - 2
        self.morph_nx = n_x

    def morph_update(self, x: np.array):
        """Morph the airfoil using the FFD points specified before.

        If symmetric:
            - len(x) = n_pts_chordwise
            - In the symmetric case, the displacement of the upper equals the
              negative displacement of the lower FFD points.
            - The number of FFD points is 2 x n_pts_chordwise, but only
              n_pts_chordwise have to be specified due to symmetry.

        Else:
            - len(x) = 2 * n_pts_chordwise - 2
            - TE and LE points are deformed symmetrically.
            - FFD assignment:
                x[-2]: LE symmetrically
                x[-1]: TE symmetrically
                x[0:-2]: other FFD points

        Parameters
        ----------
        x : np.array
            array of design variables (DV)
        """

        currentDV = self.geo.getValues()["shape"]
        newDV = currentDV.copy()

        # i -> chordwise
        # j -> normal to chord
        # k -> spanwise

        if self.morph_symmetric:
            lower_front_idx = self.geo.getLocalIndex(0)[:, 0, 0]
            lower_rear_idx = self.geo.getLocalIndex(0)[:, 0, 1]
            upper_front_idx = self.geo.getLocalIndex(0)[:, 1, 0]
            upper_rear_idx = self.geo.getLocalIndex(0)[:, 1, 1]

            j = 0
            k = 0
            for i in lower_front_idx:
                newDV[i] = x[j]
                newDV[lower_rear_idx[k]] = x[j]
                j += 1
                k += 1

            j = 0
            k = 0
            for i in upper_front_idx:
                newDV[i] = -x[j]
                newDV[upper_rear_idx[k]] = -x[j]
                j += 1
                k += 1

        if self.morph_camber_only:
            # TODO
            # i -> chordwise
            # j -> normal to chord
            # k -> spanwise

            lower_front_idx = self.geo.getLocalIndex(0)[1:-1, 0, 0]  # k = 0, "hub"
            lower_rear_idx = self.geo.getLocalIndex(0)[1:-1, 0, 1]  # k = 1, "tip"
            upper_front_idx = self.geo.getLocalIndex(0)[1:-1, 1, 0]
            upper_rear_idx = self.geo.getLocalIndex(0)[1:-1, 1, 1]

            # -------------------------
            # Other Points
            # -------------------------
            j = 0
            k = 0
            for i in lower_front_idx:
                newDV[i] = x[j]
                newDV[lower_rear_idx[k]] = x[j]
                j += 1
                k += 1

            j = 0
            k = 0
            for i in upper_front_idx:
                # newDV[i] = x[j]
                newDV[upper_rear_idx[k]] = x[j]
                j += 1
                k += 1

        else:
            # i -> chordwise
            # j -> normal to chord
            # k -> spanwise

            lower_front_idx = self.geo.getLocalIndex(0)[1:-1, 0, 0]  # k = 0, "hub"
            lower_rear_idx = self.geo.getLocalIndex(0)[1:-1, 0, 1]  # k = 1, "tip"
            upper_front_idx = self.geo.getLocalIndex(0)[1:-1, 1, 0]
            upper_rear_idx = self.geo.getLocalIndex(0)[1:-1, 1, 1]

            lower_LE_idx = self.geo.getLocalIndex(0)[0, 0, :]
            upper_LE_idx = self.geo.getLocalIndex(0)[0, 1, :]
            lower_TE_idx = self.geo.getLocalIndex(0)[-1, 0, :]
            upper_TE_idx = self.geo.getLocalIndex(0)[-1, 1, :]

            # -------------------------
            # Other Points
            # -------------------------
            j = 0
            k = 0
            for i in lower_front_idx:
                newDV[i] = x[j]
                newDV[lower_rear_idx[k]] = x[j]
                j += 1
                k += 1

            k = 0
            for i in upper_front_idx:
                newDV[i] = x[j]
                newDV[upper_rear_idx[k]] = x[j]
                j += 1
                k += 1

            # -------------------------
            # LE / TE
            # -------------------------
            if self.morph_fix_TE:
                for i in lower_LE_idx:
                    newDV[i] = x[-1]
                for i in upper_LE_idx:
                    newDV[i] = -x[-1]
            else:
                # Make LE / TE symmetric s.t. the points do not move
                for i in lower_LE_idx:
                    newDV[i] = x[-2]
                for i in upper_LE_idx:
                    newDV[i] = -x[-2]

                for i in lower_TE_idx:
                    newDV[i] = x[-1]
                for i in upper_TE_idx:
                    newDV[i] = -x[-1]

        self.geo.setDesignVars({"shape": newDV.copy()})

        data = self.geo.update("airfoil")

        self.name = f"{self.name} mod"

        self.data = data[:, 0:2]

    def save_txt(self, fname: str):
        """Save airfoil data to disc.

        Parameters
        ----------
        fname : str
            fname
        """
        np.savetxt(fname, self.data, header=self.name, comments="")

    def save_yaml(self, fname: str):
        """Save airfoil data to YAML.

        Parameters
        ----------
        fname : str
            fname
        """
        import yaml

        data = {
            "name": self.name,
            "t_max": float(self.thickness_max),
            "data": self.data.tolist(),
        }

        with open(fname, "w") as f:
            yaml.dump(data, f, default_flow_style=None)

    def save_obj(self, fname: str):
        """Save airfoil object to disc.

        Parameters
        ----------
        fname : str
            fname
        """
        with open(fname, "wb") as f:
            dump(self, f)

    def _upper_lower_split(self, data, use_half_n=True) -> tuple:
        """
        Split airfoil in upper and lower.

        Parameters
        ----------
        data : _type_
            _description_
        use_half_n : bool, optional
            _description_, by default True

        Returns
        -------
        tuple
            upper, lower
        """

        # Find LE Index
        i_LE = len(data) // 2 if use_half_n else np.argmin(data[:, 0])

        upper = data[: i_LE + 1]
        lower = data[i_LE:]
        return upper, lower

    @staticmethod
    def _get_spline(data, order, **kwargs):
        x = data[:, 0]
        y = data[:, 1]

        return splrep(x, y, k=order, **kwargs)

    @staticmethod
    def _get_coords(tck, x):
        y = splev(x, tck)
        return np.transpose(np.array([x, y]))

    def _refine(
        self,
        upper_raw,
        lower_raw,
        n: int,
        order: int = 2,
        spacing: str = "cosine",
        n_correction=10,
        smoothing=0,
        **kwargs,
    ):
        n_lower = int((n - 1) / 2) + 1
        n_upper = n_lower

        # refine upper
        x = sampling(spacing, self._chord, 0.0, n_upper, **kwargs)

        upper_raw = np.flip(upper_raw, axis=0)

        if not strictly_monotonic(upper_raw[:, 0]):
            upper_raw[-n_correction:, 0] = np.linspace(
                upper_raw[-n_correction, 0], upper_raw[-1, 0], n_correction
            )

            fig, ax = plt.subplots()
            ax.plot(upper_raw[:, 0], upper_raw[:, 1], ".-")
            ax.plot(lower_raw[:, 0], lower_raw[:, 1], ".-")
            ax.set_aspect("equal")
            plt.show()
            # raise Warning(f"upper_raw_x is not strictly monotonic! {upper_raw}")
        if not strictly_monotonic(lower_raw[:, 0]):
            lower_raw[-n_correction:, 0] = np.linspace(
                lower_raw[-n_correction, 0], lower_raw[-1, 0], n_correction
            )

            fig, ax = plt.subplots()
            ax.plot(upper_raw[:, 0], upper_raw[:, 1], ".-")
            ax.plot(lower_raw[:, 0], lower_raw[:, 1], ".-")
            ax.set_aspect("equal")
            plt.show()
            # raise Warning(f"lower_raw_x is not strictly monotonic! {lower_raw}")

        tck = self._get_spline(upper_raw, order, s=smoothing)
        upper = self._get_coords(tck, x)

        # refine lower
        x = sampling(spacing, 0.0, self._chord, n_lower, **kwargs)
        tck = self._get_spline(lower_raw, order, s=smoothing)
        lower = self._get_coords(tck, x)

        return upper, lower

    def refine(
        self,
        n: int = 241,
        order: int = 2,
        spacing: str = "cosine",
        smoothing=0,
        n_correction=10,
        **kwargs,
    ) -> None:
        """Refines airfoil using spline interpolation.

        Parameters
        ----------
        n : int
            number of points (odd number!)
        order : int
            interpolation spline order
        spacing : str, optional
            spacing method for points: "cosine" or "linear", by default "cosine"
        """
        assert n % 2, "n must be odd!"

        n_old = copy(self.n)
        # ns = np.linspace(n_old, n, 10, dtype=np.int64)

        upper_raw = self.upper
        lower_raw = self.lower

        upper, lower = self._refine(
            upper_raw,
            lower_raw,
            n=n,
            order=order,
            spacing=spacing,
            smoothing=smoothing,
            n_correction=n_correction,
            **kwargs,
        )

        # stich everything together
        data_refined = np.concatenate((upper[:-1], [[0, 0]], lower[1:]))

        self.data = data_refined
        self._unitize(order=order, spacing=spacing, n_correction=n_correction, **kwargs)
        self._recompute()

    def thicken(
        self,
        scaling_y: list = [1.0, 1.0],
        x: list = [0.0, 1.0],
        interpolation_method: str = "linear",
        n_correction=10,
    ):
        """Thickens the airfoil.

        Parameters
        ----------
        scaling_y : list, optional
            list of scaling factors, by default [1.0, 1.0]
        x : list, optional
            list of x coordinates, by default [0.0, 1.0]
        interpolation_method : str, optional
            interpolation method: ["linear", "quadratic" "cubic", "zero", ...],
            by default "linear"
        """

        x = np.array(x)
        scaling_y = np.array(scaling_y)

        scaling_t_upper = interp1d(x, scaling_y, kind=interpolation_method)(
            self.upper[:, 0]
        )
        scaling_t_lower = interp1d(x, scaling_y, kind=interpolation_method)(
            self.lower[:, 0]
        )

        y_mean_upper = self.camberline[:, 1]
        y_mean_lower = np.flip(y_mean_upper)

        y_upper = (self.upper[:, 1] - y_mean_upper) * scaling_t_upper + y_mean_upper
        y_lower = (self.lower[:, 1] - y_mean_lower) * scaling_t_lower + y_mean_lower

        upper_new = np.array([self.upper[:, 0], y_upper]).T
        lower_new = np.array([self.lower[:, 0], y_lower]).T

        self.data = np.vstack((upper_new, lower_new[1:, :]))
        self._unitize(
            order=self.order,
            spacing=self.spacing,
            n_correction=n_correction,
            **self.kwargs,
        )
        self._recompute()

    def plot(self, fname=None, dpi=300, show=False, legend=False, ffd=True):
        """Plots the airfoil.

        Parameters
        ----------
        fname : str, optional
            Filename of image to save, by default None
        dpi : int, optional
            DPI of png file, by default 300
        plot_show : bool, optional
        """
        fig, ax = plt.subplots()

        # Figure Sizing
        fct_size = 1.0 / np.abs(self.y.max() - self.y.min())
        width = 18 / 2.54
        space = 1.0 if fct_size > 5 else 0.5
        height = width / fct_size + min(1, space)
        fig.set_size_inches(width, height)

        ax.plot(
            self.upper[:, 0],
            self.upper[:, 1],
            ".-",
            lw=1,
            c="tab:red",
            s=2,
            label="upper",
        )
        ax.plot(
            self.lower[:, 0],
            self.lower[:, 1],
            ".-",
            lw=1,
            c="tab:blue",
            s=2,
            label="lower",
        )
        ax.plot(
            self.camberline[:, 0],
            self.camberline[:, 1],
            "--",
            lw=0.5,
            c="tab:green",
            label="camber line",
        )
        ax.plot(
            self.chordline[:, 0],
            self.chordline[:, 1],
            "--",
            lw=0.5,
            c="k",
            label="chord line",
        )
        if ffd:
            try:
                ni, nj, _ = self.pts_ffd_2d.shape
                for i in range(ni):
                    for j in range(nj):
                        ax.plot(
                            self.pts_ffd_2d[i, j, 0],
                            self.pts_ffd_2d[i, j, 1],
                            ".",
                            color="k",
                        )
            except:
                pass

        ax.set(
            aspect="equal",
            xlabel=r"$x$",
            ylabel=r"$y$",
            title=self.name,
            xlim=[-0.01 * self._chord, self._chord * 1.01],
        )
        if legend:
            ax.legend()
        if show == True:
            plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)

    def _unitize(self, order=2, spacing="cosine", n_correction=10, **kwargs):
        """De-rotates the airfoil."""
        self._recompute()

        n_0 = self.n

        # Spline Interpolation
        n_pts_fine = 20 * self.n
        t_eval = np.linspace(self.s_airfoil.start, self.s_airfoil.end, n_pts_fine)
        airfoil = self.s_airfoil.evaluate(t_eval)

        TE = self._find_TE(airfoil)
        LE = self._find_LE(airfoil, TE)[0]

        chord = np.linalg.norm(LE - TE, axis=0)
        self._chord = chord

        # Move LE to Origin
        # airfoil = self.data
        airfoil = airfoil - LE
        TE = TE - LE
        LE = np.array([0, 0])

        dy = TE[1] - LE[1]
        dx = TE[0] - LE[0]
        twist = -np.arctan2(dy, dx)

        # Rotate the Airfoil
        R = np.array(
            [
                [np.cos(twist), -np.sin(twist)],
                [np.sin(twist), np.cos(twist)],
            ]
        )
        for i in range(len(airfoil)):
            airfoil[i, :] = R.dot(airfoil[i, :])

        TE_new = (airfoil[0, :] + airfoil[-1, :]) / 2.0

        # Scale airfoil
        s = 1 / TE_new[0]
        airfoil *= s * chord

        # Re-Tesselate Airfoil
        upper, lower = self._upper_lower_split(airfoil, use_half_n=False)
        upper, lower = self._refine(
            upper,
            lower,
            n=n_0,
            order=order,
            spacing=spacing,
            n_correction=n_correction,
            **kwargs,
        )
        data_refined = np.concatenate((upper[:-1], [[0, 0]], lower[1:]))
        self.data = data_refined

        return chord, twist

    def fit_cst(self, order=12):
        """Fits CST parameterization to airfoil.

        Parameters
        ----------
        order : int, optional
            number of parameters - 1, by default 12

        Returns
        -------
        tuple
            a_ca : float
                parameters for camber line
            a_th : float
                parameters for thickness distribution
            t_te : float
                thickness of TE
            data : array
                airfoil data
        """

        # read profile coordinates
        upper = self.upper
        lower = self.lower

        # transpose stuff
        upperT = np.flip(np.transpose(upper), axis=1)
        lowerT = np.transpose(lower)

        # prepare coords2cst
        idx = np.argmin(self.data[:, 0])
        coords_upper = np.flipud(self.data[: idx + 1])
        coords_lower = self.data[idx:]
        x = (1 - np.cos(np.linspace(0, 1) * np.pi)) / 2
        y_u = np.interp(x, coords_upper[:, 0], coords_upper[:, 1])
        y_l = np.interp(x, coords_lower[:, 0], coords_lower[:, 1])

        a_ca, a_th, t_te = coords2cst(x, y_u, y_l, order, order)

        # cst2coords
        x_sm_u, y_u_sm_u, _, _, _ = cst2coords(a_ca, a_th, t_te, len(upper))
        x_sm_l, _, y_u_sm_l, _, _ = cst2coords(a_ca, a_th, t_te, len(lower))

        # put everything together
        finished_x = np.concatenate((np.flip(x_sm_u)[:-1], x_sm_l))
        finished_y = np.concatenate((np.flip(y_u_sm_u)[:-1], y_u_sm_l))
        finished = np.transpose(np.array([finished_x, finished_y]))

        return a_ca, a_th, t_te, finished

    def close_TE(self) -> None:
        """Closes the TE to get a numerically clean airfoil."""
        if self.upper[0, 0] != self._chord or self.lower[0, 1] != 0.0:
            self.data[0, 0] = self._chord
            self.data[0, 1] = 0.0
        if self.lower[-1, 0] != self._chord or self.lower[-1, 1] != 0.0:
            self.data[-1, 0] = self._chord
            self.data[-1, 1] = 0.0

    def round_TE(
        self, n_pts=20, distance=0.4, order=4, check_geometry=True, n_correction=10
    ):
        """Rounds and closes the TE using a spline. Initial airfoil must be blunt!

        Parameters
        ----------
        n_pts : int, optional
            number of points, by default 20
        distance : float, optional
            normalized by TE gap, by default 0.4
        order : int, optional
            spline order, by default 4

        References
        ----------
        https://github.com/mdolab/prefoil/blob/c59f087134517f2d4a9f1a621284da1f3abf7a67/prefoil/airfoil.py#L12:~:text=def%20roundTE(self%2C%20xCut%3D0.98%2C%20k%3D4%2C%20nPts%3D20%2C%20dist%3D0.4)%3A
        """

        # Length for Making Rounded TE
        dx = self.thickness_te * distance

        # Knot Vector
        t = [0] * order + [0.5] + [1] * order

        # Control Points
        coeff = np.zeros((order + 1, 2))
        for ii in [0, -1]:
            coeff[ii] = self.s_airfoil.evaluate(np.abs(ii))
            dX_ds = self.s_airfoil.derivative(np.abs(ii))
            dy_dx = dX_ds[1] / dX_ds[0]

            coeff[3 * ii + 1] = np.array(
                [coeff[ii, 0] + dx * 0.5, coeff[ii, 1] + dy_dx * dx * 0.5]
            )
            # ii = 0 -> coeff[1] and ii = -1 -> coeff[-2]
        if order == 4:
            chord = self.TE - self.LE
            chord /= np.linalg.norm(chord)
            coeff[2] = np.array(
                [self.TE[0] + chord[0] * dx, self.TE[1] + chord[1] * dx]
            )

        # Make the TE Curve
        # te_curve = cf.Curve(
        #     basis=cf.BSplineBasis(order=order, knots=t), controlpoints=coeff
        # ).reparam()

        te_curve = Spline(coeff)

        # Combine Curves
        upper_curve, lower_curve = te_curve.split(0.5)
        t_lower = np.linspace(1.0, 0.5, n_pts)
        t_upper = np.linspace(0.5, 0.0, n_pts)
        pts_lower = lower_curve.evaluate(t_lower)
        pts_upper = upper_curve.evaluate(t_upper)

        if check_geometry:
            assert np.any(
                pts_upper[:, 1] > pts_lower[:, 1]
            ), "Distance too big! (upper < lower)"

            upper_derivatves = upper_curve.derivative(np.linspace(0.0, 0.5, n_pts))
            dy_dx_upper = upper_derivatves[:, 1] / upper_derivatves[:, 0]

            assert strictly_decreasing(
                dy_dx_upper[:-1].tolist()
            ), "Distance too big! (upper derivative not monotonically decreasing)"

        coords = np.vstack(
            (pts_upper, self.upper[1:, :], self.lower[:-1, :], pts_lower)
        )

        self.data = coords
        self._unitize(
            order=self.order,
            spacing=self.spacing,
            n_correction=n_correction,
            **self.kwargs,
        )
        self._recompute()

    def add_TE_thickness(self, t_TE: float, chord_split: float = 0.5, n_correction=10):
        """Cuts/extends the TE to achieve the required TE thickness.

        Parameters
        ----------
        t_TE : float
            required TE thickness
        """

        lower = self.lower
        upper = self.upper

        t = np.abs(-np.flip(lower[:, 1], axis=0) + upper[:, 1])

        n_t = len(t)

        i_split = int(n_t * chord_split)
        t_search = t[:i_split]
        x = upper[:, 0]
        x_search = upper[:i_split, 0]
        n_t_search = len(t_search)

        f = interp1d(t_search, x_search, kind="linear", fill_value="extrapolate")
        x_TE = f(t_TE)

        mask = t_search > t_TE
        x_new = np.insert(x_search[mask], 0, x_TE)

        x_res = np.hstack((x_new, x[i_split:]))
        f = interp1d(x, upper[:, 1], kind="linear", fill_value="extrapolate")
        y_upper = f(x_res)

        f = interp1d(
            x, np.flip(lower[:, 1], axis=0), kind="linear", fill_value="extrapolate"
        )
        y_lower = f(x_res)

        upper_new = np.array([x_res, y_upper]).T
        lower_new = np.flip(np.array([x_res, y_lower]).T, axis=0)

        upper_new[:, 0] /= x_TE
        lower_new[:, 0] /= x_TE

        self.data = np.vstack((upper_new, lower_new[1:, :])) * self._chord

        TE = (upper_new[0, :] + lower_new[-1, :]) / 2.0
        self._unitize(
            order=self.order,
            spacing=self.spacing,
            n_correction=n_correction,
            **self.kwargs,
        )
        self._recompute()

    def section_analysis(self, mesh_size=None, plot=False):
        """Calculate the mechanical 2D section properties.

        Returns
        -------

        section: Section object

        See https://sectionproperties.readthedocs.io/en/latest/rst/post.html
        """

        try:
            from sectionproperties.pre.geometry import Geometry
            from sectionproperties.analysis.section import Section
        except:
            raise Exception(
                "Please install sectionproperties>=2.1.3 (needs Python<=3.10)!"
            )

        # ------------------------------
        # Create Shape
        # ------------------------------

        points = self.data.tolist()
        facets = []

        for i in range(self.n):
            _list = [i, 0] if i == (self.n - 1) else [i, i + 1]
            facets.append(_list)

        n_control_pt = self.n // 4
        pt_control = self.lower[n_control_pt] + self.upper[n_control_pt]

        control_points = [pt_control.tolist()]

        geometry = Geometry.from_points(points, facets, control_points)

        # ------------------------------
        # Meshing
        # ------------------------------

        if mesh_size is None:
            mesh_size = self.thickness_max / 10
        geometry.create_mesh(mesh_sizes=mesh_size)
        section = Section(geometry)
        section.plot_mesh() if plot else None  # plot the generated mesh

        # ------------------------------
        # Analysis
        # ------------------------------

        section.calculate_geometric_properties()
        section.calculate_warping_properties()
        section.calculate_plastic_properties()

        section.plot_centroids() if plot else None

        # ------------------------------
        # Results
        # ------------------------------

        return section

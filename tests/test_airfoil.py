from re import A
import unittest
from airfoil.airfoil import Airfoil
import proplot as pplt
from copy import copy
import matplotlib.pyplot as plt
import numpy as np


class TestAirfoil(unittest.TestCase):
    def test_load_txt(self):
        try:
            af = Airfoil.load_txt("../data/naca6503.dat")
            af.plot()
        except Exception as e:
            self.fail(f"test_load_txt failed: {e}")

    def test_circular_TE(self):
        try:
            af = Airfoil.load_txt("../data/af_mid.dat")
            af.round_TE(n_pts=5, distance=0.2)

        except Exception as e:
            self.fail(f"test_circular_TE failed: {e}")

    def test_wennerstrom(self):
        try:
            af = Airfoil.wennerstrom(241)
        except Exception as e:
            self.fail(f"test_wennerstrom failed: {e}")

    def test_naca(self):
        try:
            af = Airfoil.naca("0012", n=241, chord=1)
            af.add_TE_thickness(0.01)
            af.round_TE()
            af.plot(show=False)
        except Exception as e:
            self.fail(f"test_naca failed: {e}")

    def test_thicken(self):
        try:
            factor = 1.15
            af = Airfoil.load_txt("../data/af_mid.dat")
            af.refine(251)
            af.plot(show=False)
            af.add_TE_thickness(0.005)
            af_scaled = copy(af)
            af_scaled.thicken(
                [1.3, 1.3, 1.3], [1, 0.5, 0.0], interpolation_method="quadratic"
            )

            af.plot_airfoils([af, af_scaled], show=False)
        except Exception as e:
            self.fail(f"test_naca failed: {e}")

    def test_morph(self):
        try:
            af_init = Airfoil.load_txt("../data/af_mid.dat")
            ymargin_left = 0.1
            ymargin_right = 0.1
            lb = -0.05
            ub = 0.05

            af_init.plot()
            af = copy(af_init)
            n_pts_chordwise = 4

            af.ffd(
                n_pts_chordwise,
                fname="ffd.xyz",
                ymargin_left=ymargin_left,
                ymargin_right=ymargin_right,
            )

            af.morph_init(
                fname_ffd="ffd.xyz",
            )

            x = np.random.uniform(low=lb, high=ub, size=(n_pts_chordwise * 2 - 2,))
            af.morph_update(x)
            af.plot_airfoils([af_init, af], legend=False, show=True)

        except Exception as e:
            self.fail(f"test_naca failed: {e}")

    def test_morph_camber_only(self):
        try:
            af_init = Airfoil.load_txt("../data/af_mid.dat")
            ymargin_left = 0.1
            ymargin_right = 0.1
            lb = -0.05
            ub = 0.05

            af_init.plot()
            af = copy(af_init)
            n_pts_chordwise = 4

            af.ffd(
                n_pts_chordwise,
                fname="ffd.xyz",
                ymargin_left=ymargin_left,
                ymargin_right=ymargin_right,
            )

            camber_only = True
            af.morph_init(
                fname_ffd="ffd.xyz",
                camber_only=camber_only,
            )

            if camber_only:
                x = np.random.uniform(low=lb, high=ub, size=(n_pts_chordwise - 2,))
            else:
                x = np.random.uniform(low=lb, high=ub, size=(n_pts_chordwise * 2 - 2,))
            af.morph_update(x)
            af.plot_airfoils([af_init, af], legend=False, show=True)

        except Exception as e:
            self.fail(f"test_naca failed: {e}")

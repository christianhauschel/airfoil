from re import A
import unittest
from airfoil.airfoil import Airfoil
import proplot as pplt
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tempfile


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

    def test_normalize_exact_le_te(self):
        af = Airfoil.naca("0012", n=241, chord=3, finite_TE=True)
        angle = np.radians(17.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        af.data = af.data @ rotation.T + np.array([0.3, -0.2])

        af.normalize()

        np.testing.assert_array_equal(af.LE, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(af.TE, np.array([1.0, 0.0]))
        self.assertEqual(af.chord, 1.0)

    def test_selig_lednicer_format_conversion(self):
        selig = np.array(
            [
                [1.0, 0.01],
                [0.5, 0.05],
                [0.0, 0.0],
                [0.5, -0.04],
                [1.0, -0.01],
            ]
        )

        upper, lower = Airfoil._selig_to_lednicer_data(selig)
        np.testing.assert_array_equal(
            upper, np.array([[0.0, 0.0], [0.5, 0.05], [1.0, 0.01]])
        )
        np.testing.assert_array_equal(
            lower, np.array([[0.0, 0.0], [0.5, -0.04], [1.0, -0.01]])
        )
        np.testing.assert_array_equal(
            Airfoil._lednicer_to_selig_data(upper, lower), selig
        )

    def test_detect_lednicer_format(self):
        lines = [
            "Test Airfoil\n",
            "3 3\n",
            "\n",
            "0.0 0.0\n",
            "0.5 0.05\n",
            "1.0 0.01\n",
            "\n",
            "0.0 0.0\n",
            "0.5 -0.04\n",
            "1.0 -0.01\n",
        ]
        self.assertEqual(Airfoil._detect_txt_format(lines, skiprows=1), "lednicer")
        self.assertEqual(Airfoil._detect_txt_format(lines, skiprows=2), "lednicer")

    def test_save_txt_defaults_to_selig(self):
        af = Airfoil.__new__(Airfoil)
        af.name = "Format Test"
        af.data = np.array(
            [
                [1.0, 0.01],
                [0.5, 0.05],
                [0.0, 0.0],
                [0.5, -0.04],
                [1.0, -0.01],
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = Path(tmpdir) / "airfoil.dat"
            af.save_txt(fname)
            lines = fname.read_text().splitlines()

        self.assertEqual(lines[0], "Format Test")
        self.assertEqual(len(lines), 6)
        self.assertEqual(float(lines[1].split()[0]), 1.0)

    def test_refine_preserves_le_te_with_sqrt_smoothing(self):
        af = Airfoil.__new__(Airfoil)
        af._chord = 1.0

        upper_raw = np.array(
            [[1.0, 0.0], [0.25, 0.08], [0.02, 0.025], [0.0, 0.0]]
        )
        lower_raw = np.array(
            [[0.0, 0.0], [0.02, -0.02], [0.25, -0.06], [1.0, 0.0]]
        )

        upper, lower = af._refine(
            upper_raw, lower_raw, n=41, order=3, smoothing=True
        )

        np.testing.assert_array_equal(upper[-1], np.array([0.0, 0.0]))
        np.testing.assert_array_equal(lower[0], np.array([0.0, 0.0]))
        self.assertEqual(upper[0, 0], 1.0)
        self.assertEqual(lower[-1, 0], 1.0)
        self.assertTrue(np.all(np.diff(upper[:, 0]) <= 0.0))
        self.assertTrue(np.all(np.diff(lower[:, 0]) >= 0.0))
        self.assertEqual(Airfoil._smoothing_strength(True), 0.25)

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

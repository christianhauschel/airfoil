"""Example which shows the interpolation of a list of airfoils."""

# %%

from matplotlib import tight_layout
from airfoil import Airfoil, interpolate_airfoils
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# create list of random naca4 numbers
names = [
    "4515",
    "5408",
    "6305",
]

afs = [Airfoil.naca(name) for name in names]

afs_int = interpolate_airfoils(afs, plot_debug=True, kind="linear")

# get chord of all interpolated airfoils
chords = [af.chord for af in afs_int]

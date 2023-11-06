# %% Imports

from airfoil import Airfoil

# %% Example 1: NACA4 and CSV-Files

af = Airfoil.naca("4412", finite_TE=True)
af.normalize()
af.plot(show=True)

af.save_csv(f"data/{af.name}.csv")

af1 = Airfoil.load_csv(f"data/{af.name}.csv")
af1.plot(show=True)

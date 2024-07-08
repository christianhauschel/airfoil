# %% Imports

from airfoil import Airfoil
import numpy as np

# %% Example 1: NACA4 and CSV-Files

af = Airfoil.naca("4412", finite_TE=True)
af.normalize()
# af.plot(show=True)

af.save_csv(f"data/{af.name}.csv")

af1 = Airfoil.load_csv(f"data/{af.name}.csv")
# af1.plot(show=True)

af1.add_TE_thickness(0.03)
# af1.rotate(np.radians(20))


af1.round_TE()
# print(af1.TE)
af1.normalize()
af1.refine(241)
af1.plot(show=True, fname="test.svg") 
# af1.plot(show=False, fname="test.svg") 


# %%
# %%

from airfoil.geometry import *
from airfoil.airfoil import Airfoil

# %% NACA

af = Airfoil.naca("6510", n=51)
af.plot(show=True)
af.add_TE_thickness(0.01)
af.circular_TE()
af.plot(show=True)

# %% CST

a_ca, a_th, t_te, data = af.fit_cst(order=12)
af1 = Airfoil.cst(a_ca, a_th, t_te, 241)
af1.plot()

# %% PARSEC

af2 = Airfoil.parsec(241)
af2.plot()
af2.circular_TE()
af2.plot()

# %% Wennerstrom

af3 = Airfoil.wennerstrom(241)
af3.plot("airfoil.png")

print("t", af3.thickness)
print("t_max", af3.thickness_max)

# %% Section Analysis

section = af3.section_analysis(plot=True)


# %% Results

section.display_results()
print(f"J: {section.get_j():0.1e}")

# %% RUN ALL

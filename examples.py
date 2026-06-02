# %% Imports

from airfoil import Airfoil, plot_airfoils
from copy import copy

af_base = Airfoil.load_txt("data/fauvel_14_base.dat")

af_refined = copy(af_base)
af_refined.refine(n=241, smoothing=0.01)
af_refined.normalize()

af_refined.name = "Fauvel 14 (refined)"
af_refined.save_txt("data/fauvel_14_refined.dat")

af_refined_TE = copy(af_refined)

af_refined_TE.add_TE_thickness(0.004)
af_refined_TE.normalize()
af_refined_TE.name = "Fauvel 14 (refined, TE=0.004)"

plot_airfoils([af_base, af_refined, af_refined_TE], show=True)


af_refined_TE.save_txt("data/fauvel_14_refined_TE.dat")


# %%

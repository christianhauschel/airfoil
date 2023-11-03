# %%


from airfoil.airfoil import Airfoil

# %% Load

af = Airfoil.load_txt("af_mid.dat")
# af = Airfoil.load_txt("naca6503.dat")
# af.plot(show=True)

af.circular_TE()
af.plot()

# %% Refine

af.refine(101)

af.plot(show=True)


# %%

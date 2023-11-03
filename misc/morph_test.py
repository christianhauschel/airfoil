# %% Modules

import argparse
from airfoil import Airfoil
import numpy as np
import proplot as pplt
from pathlib import Path
import yaml
from copy import copy

# %% Parameters

dir_out = Path("./data")

camber_only = True
fix_TE = False
symmetric = False

# af_init = Airfoil.load_txt(dir_out / "af_mid.dat")
# ymargin_left = 0.04
# ymargin_right = 0.03
# lb = 0.0
# ub = 0.07

# af_init = Airfoil.load_txt(dir_out /"af_tip.dat")
# ymargin_left = 0.036
# ymargin_right = 0.03
# lb = 0.00
# ub = 0.05

af_init = Airfoil.load_txt(dir_out / "af_hub.dat")
ymargin_left = 0.1
ymargin_right = 0.1
lb = -0.07
ub = 0.07

n_pts_chordwise = 4

# %% FFD


af_init.plot()
af = copy(af_init)


af.ffd(
    n_pts_chordwise,
    fname="ffd.xyz",
    ymargin_left=ymargin_left,
    ymargin_right=ymargin_right,
)
af.plot()

# %% Morph

af.morph_init(
    fname_ffd="ffd.xyz",
    camber_only=camber_only,
    symmetric=symmetric,
    fix_TE=fix_TE,
)

for _ in range(4):
    if camber_only:
        x = np.random.uniform(low=lb, high=ub, size=(n_pts_chordwise - 2,))
    else:
        if fix_TE:
            x = np.random.uniform(low=lb, high=ub, size=(n_pts_chordwise * 2 - 3,))
        else:
            x = np.random.uniform(low=lb, high=ub, size=(n_pts_chordwise * 2 - 2,))
    af.morph_update(x)
    af.plot_airfoils([af_init, af], legend=False)


# %% RUN ALL

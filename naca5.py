# %%

from matplotlib import tight_layout
import numpy as np

def naca5(Ides, Xx, Yt, Yc, n, Xb, Yb, Nb):
    # TE point bunching parameter
    an = 1.5
    digits = '0123456789'

    n5 = Ides // 10000
    n4 = (Ides - n5 * 10000) // 1000
    n3 = (Ides - n5 * 10000 - n4 * 1000) // 100
    n2 = (Ides - n5 * 10000 - n4 * 1000 - n3 * 100) // 10
    n1 = Ides - n5 * 10000 - n4 * 1000 - n3 * 100 - n2 * 10

    n543 = 100 * n5 + 10 * n4 + n3

    if n543 == 210:
        m, c = 0.0580, 361.4
    elif n543 == 220:
        m, c = 0.1260, 51.64
    elif n543 == 230:
        m, c = 0.2025, 15.957
    elif n543 == 240:
        m, c = 0.2900, 6.643
    elif n543 == 250:
        m, c = 0.3910, 3.230
    else:
        print('Illegal 5-digit designation')
        print('First three digits must be 210, 220, ... 250')
        Ides = 0
        return

    t = (n2 * 10 + n1) / 100.0
    anp = an + 1.0

    for i in range(n):
        frac = i / (n - 1)
        if i == n - 1:
            Xx[i] = 1.0
        else:
            Xx[i] = 1.0 - anp * frac * (1.0 - frac)**an - (1.0 - frac)**anp

        Yt[i] = (0.29690 * np.sqrt(Xx[i]) - 0.12600 * Xx[i] - 0.35160 * Xx[i]**2 +
                 0.28430 * Xx[i]**3 - 0.10150 * Xx[i]**4) * t / 0.20

        if Xx[i] < m:
            Yc[i] = (c / 6.0) * (Xx[i]**3 - 3.0 * m * Xx[i]**2 + m**2 * (3.0 - m) * Xx[i])
        else:
            Yc[i] = (c / 6.0) * m**3 * (1.0 - Xx[i])

    ib = 0
    for i in range(n - 1, -1, -1):
        ib += 1
        Xb[ib] = Xx[i]
        Yb[ib] = Yc[i] + Yt[i]

    for i in range(1, n):
        ib += 1
        Xb[ib] = Xx[i]
        Yb[ib] = Yc[i] - Yt[i]

    Nb[0] = ib

    name = 'NACA' + digits[n5] + digits[n4] + digits[n3] + digits[n2] + digits[n1]

    return name

# Example usage:
Nside = 100
Ides = 24012
Xx = np.zeros(Nside)
Yt = np.zeros(Nside)
Yc = np.zeros(Nside)
Xb = np.zeros(2 * Nside)
Yb = np.zeros(2 * Nside)
Nb = np.zeros(1)
Name = f"NACA{Ides}"

naca5(Ides, Xx, Yt, Yc, Nside, Xb, Yb, Nb, Name)

import proplot as pplt
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,1.8), tight_layout=True)
ax.plot(Xb[1:], Yb[1:])
ax.plot(Xx, Yc, "--")
ax.axis('equal')
ax.grid(True)
ax.set_title(Name)

# %%
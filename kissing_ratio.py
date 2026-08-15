import numpy as np
import matplotlib.pyplot as plt

fname = "/home/akash/hoomd-results/kissing_curve_data.csv"

c = []
s = []

L  = 0.0
rl = 0.5

for cr in np.arange(0.5, 1.02, 0.05):
    q  = np.sqrt(1 - (cr * cr) / 3)
    sz = (cr * np.sqrt(1 / 6) + q - L / rl)
    s.append(sz)
    c.append(cr)

K = np.column_stack((s, c))
np.savetxt(fname, K, delimiter=",", header="size_ratio,compression_ratio", comments="")

plt.plot(s, c, 'o-')
plt.xlim(1.14, 1.24)
plt.xlabel("Size Ratio")
plt.ylabel("Compression Ratio")
plt.title("Kissing Curve")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
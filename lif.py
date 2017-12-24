import matplotlib.pyplot as plt
import numpy as np


def lif(_x, offset=0, scale=1, k=1, t=0.5):
    diff = (np.exp(-t*k)*scale+offset) - (-np.exp(-t*k)*scale+offset)
    if _x <= t:
        val = -np.exp(-_x*k)*scale+offset+diff/2
    else:
        val = np.exp(-_x*k)*scale+offset-diff/2
    return val + diff/2


x = np.linspace(0.0, 1.0, 100)
y = [lif(_x) for _x in x]
print(y[-1])
plt.plot(x, y)
plt.show()
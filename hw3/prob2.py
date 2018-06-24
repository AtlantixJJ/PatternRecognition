"""
Plot distribution of Bornoulli
"""
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.plot(range(10))
plt.close()

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

theta = np.linspace(0, 1, 100)
plt.plot(theta, 2 * (1 - theta))
plt.plot(theta, 2 * theta)
plt.legend(["x=0", "x=1"])
plt.xlabel("theta")
plt.ylabel("probability")
plt.savefig("fig/prob2.png")
plt.close()
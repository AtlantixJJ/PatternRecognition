import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 1000)

def gaussian(x, mu=0, sigma=1):
    """
    Compute normal function
    Args:
    x   : vector
    mu  : vector
    """
    return np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def weight_gaussian(x, weight, mu=0, sigma=1):
    """
    Compute normal function
    Args:
    x   : vector
    mu  : vector
    """
    return weight * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def risk_func(x, r=1.0, s=4.0):
    return (1 - r/s) * (0.5 * gaussian(x, 1, 1) + 0.5 * gaussian(x, -1, 1))

def risk_func2(x, r=1.0, s=4.0):
    return (1 - r/s) * (1/3. * gaussian(x, 1, 1) + 2/3. * gaussian(x, 0, 1/4.))

def max_func(x, *arg):
    return max(0.5 * gaussian(x, 1, 1), 0.5 * gaussian(x, -1, 1), risk_func(x, *arg))

def max_func2(x, *arg):
    return max(1/3. * gaussian(x, 1, 1), 2/3. * gaussian(x, 0, 1/4.), risk_func2(x, *arg))

def map_func(x, func, *arg):
    return np.array([func(t, *arg) for t in x])

plt.plot(x, map_func(x, weight_gaussian, 0.5, 1, 1), '--')
plt.plot(x, map_func(x, weight_gaussian, 0.5, -1, 1), '--')
plt.plot(x, map_func(x, risk_func), '--')
plt.plot(x, map_func(x, max_func))
plt.legend(["w1", "w2", "risk", "g(x)"])
plt.savefig("fig/prob4.png")
plt.close()

plt.plot(x, map_func(x, risk_func, 0.0, 1.0))
plt.plot(x, map_func(x, risk_func, 1.0, 16.0))
plt.plot(x, map_func(x, risk_func, 1.0, 3.0))
plt.plot(x, map_func(x, risk_func, 0.8, 1.0))
plt.legend(["0", "1/16", "1/3", "0.8"])
plt.savefig("fig/prob4_risk.png")
plt.close()

plt.plot(x, map_func(x, max_func, 0.0, 1.0))
plt.plot(x, map_func(x, max_func, 1.0, 16.0))
plt.plot(x, map_func(x, max_func, 1.0, 3.0))
plt.plot(x, map_func(x, max_func, 0.8, 1.0))
plt.legend(["0", "1/16", "1/3", "0.8"])
plt.savefig("fig/prob4_total.png")
plt.close()

plt.plot(x, map_func(x, weight_gaussian, 1/3., 1, 1), '--')
plt.plot(x, map_func(x, weight_gaussian, 2/3., 0, 1/4.), '--')
plt.plot(x, map_func(x, risk_func2, 1.0, 2.0), '--')
plt.plot(x, map_func(x, max_func2))
plt.legend(["w1", "w2", "risk", "g(x)"])
plt.savefig("fig/prob4_2.png")
plt.close()
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 3, 1000)

def func1(x):
    return max(0, 1 - abs(x))

def func2(x):
    return max(0, (0.5 - abs(x - 0.5))/0.25)

def func3(x):
    return max(0, 1 - abs(x - 1))

def max_func(x):
    return max(func1(x), func2(x), func3(x))

def min_risk(x):
    if x < -1 or x > 2:
        return 0
    return 1 - max_func(x)

def map_func(x, func):
    return np.array([func(t) for t in x])

plt.plot(x, map_func(x, func1), '--')
plt.plot(x, map_func(x, func2), '--')
plt.plot(x, map_func(x, func3), '--')
plt.plot(x, map_func(x, max_func))
plt.legend(["w1", "w2", "w3", "P(f(x)|x)"])
plt.savefig("fig/prob2.png")
plt.close()

plt.plot(x, map_func(x, min_risk))
plt.savefig("fig/prob2_minrisk.png")
plt.close()
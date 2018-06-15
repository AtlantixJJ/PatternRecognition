"""
Experiment for Problem1: params = (\mu_1, \mu_2, \signma_1, \signma_2)
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def get_dataset(params=[]):
    """
    create dataset: sample and label
    """
    data_c1 = np.random.normal(params[0], params[2], (100, 2))
    data_c2 = np.random.normal(params[1], params[3], (100, 2))
    data = np.concatenate((data_c1, data_c2))
    label = np.concatenate((np.zeros((100,)), np.ones((100,))))
    return data, label

def guassian(x, mu=0, sigma=1):
    """
    Compute normal function
    Args:
    x   : vector
    mu  : vector
    """
    return np.exp(- np.dot(x - mu, x - mu) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

def get_posterior(params):
    def p_x_c1_f(x):
        return guassian(x, np.array([params[0], params[0]]), params[2])
    def p_x_c2_f(x):
        return guassian(x, np.array([params[1], params[1]]), params[3])
    return p_x_c1_f, p_x_c2_f

def eval(data, func):
    return np.array([func(v) for v in data])

def minerr_cls(p1, p2):
    """
    Minimimal error Bayesian classifier
    Args:
    p1, p2: array of feature prob.
    """
    metric = p1 / p2
    res = np.zeros_like(p1)
    res[metric > 1] = 0
    res[metric <= 1] = 1
    return res

def result2color(label):
    l = []
    for v in label:
        if v == 0:
            l.append("r")
        else:
            l.append("b")
    return l

#params = [1, 1.5, np.sqrt(0.2), np.sqrt(0.2)]
params = [1, 3, np.sqrt(0.2), np.sqrt(0.2)]

data, label = get_dataset(params)

# draw classification surface p(x|c1)
p_x_c1, p_x_c2 = get_posterior(params)
sample_p_x_c1 = eval(data, p_x_c1)
sample_p_x_c2 = eval(data, p_x_c2)
res1 = minerr_cls(sample_p_x_c1, sample_p_x_c2)
colors = result2color(res1)

acc = (res1 == label).sum()
print("Total acc:\t%f" % (float(acc) / res1.shape[0]))
acc = (res1[:100] == label[:100]).sum()
print("Omega 1:\t%f" % (float(acc) / 100.))
acc = (res1[100:] == label[100:]).sum()
print("Omega 2:\t%f" % (float(acc) / 100.))

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter3D(data[:, 0], data[:, 1], sample_p_x_c1, c=colors)
ax = fig.add_subplot(122, projection='3d')
ax.scatter3D(data[:, 0], data[:, 1], sample_p_x_c2, c=colors)
fig.savefig("MinerrorSurface_3.png")
fig.clear()
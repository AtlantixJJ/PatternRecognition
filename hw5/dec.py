import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.cluster

def default_data():
    A = [[1, 1, 1], [1, 1, 1]]
    B = [[3, 3, 3], [2, 3, 4]]
    C = [[7, 8, 9], [6, 6, 9]]
    
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    return A, B, C

def generate_data(pdfs, samples):
    """
    Mix pdfs generator with sample numbers
    """
    data = []
    for p, s in zip(pdfs, samples):
        class_data = []
        for i in range(p.shape[1]):
            print(p[0][i], p[1][i])
            class_data.append(np.random.normal(p[0][i], p[1][i], (s, 1)))
        class_data = np.concatenate(class_data, axis=1)
        print(class_data.mean(axis=0))
        data.append(class_data)
    data = np.concatenate(data, axis=0)
    return data

def scatter3d(data, name="dataset"):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1)
    plt.savefig(name + ".png")
    plt.close()

def printable(data, N=3):
    for i in range(data.shape[0]):
        print("|%.5f|%.5f|%.5f|" % (data[i][0], data[i][1], data[i][2]))

def KMeans(data, N):
    print("%d clusters" % N)
    KMeansClustor = sklearn.cluster.KMeans(n_clusters = N)
    res = KMeansClustor.fit(data)
    print("Cluster centers: ")
    printable(res.cluster_centers_)
    labels = res.labels_
    print("Cluster sigma: ")
    std = []
    for i in range(0, N, 1):
        mask = (labels == i)
        std.append(data[mask].std(axis=0))
    std = np.array(std)
    printable(std)

    return np.array(res.cluster_centers_), std

def guassian(x, mu=0, sigma=1):
    """
    Compute normal function
    Args:
    x   : vector
    mu  : vector
    """
    return np.exp(- (x - mu) ** 2 / (2 * sigma ** 2) ) / (np.sqrt(2 * np.pi) * sigma)
    #return np.exp(- np.dot(x - mu, x - mu) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

def get_fi(x, ind, t_i, theta_i):
    assert(0 <= ind and ind <=2)

    return guassian(x, theta_i[ind][0], theta_i[ind][1]) * t_i[ind]

def get_f(x, t_i, theta_i, N):
    return sum([t_i[i] * get_fi(x, i, t_i, theta_i) for i in range(N)])

def dataset_loss(data, t_i, theta_i, N):
    return sum([np.log(get_f(x, t_i, theta_i, N))  for x in data])

def derivative_t(data, t_i, theta_i, N):
    dldt = np.zeros((N,), dtype="float32")
    for x in data:
        fi = [get_fi(x, i, t_i, theta_i) for i in range(N)]
        f = sum([t_i[i] * fi[i] for i in range(N)])
        for i in range(N):
            dldt[i] += fi[i]/f
    return dldt / data.shape[0]

def derivative_theta(data, t_i, theta_i, N=3):
    dldo = np.zeros((N, 2), dtype="float32")

    for x in data:
        fi = [get_fi(x, i, t_i, theta_i) for i in range(N)]
        f = sum([t_i[i] * fi[i] for i in range(N)])
        for i in range(N):
            t = t_i[i] / f * fi[i]
            
            dldo[i][0] += t * (x - theta_i[i][0]) / theta_i[i][1] ** 2
            dldo[i][1] += t * (x - theta_i[i][0]) ** 2 / 4 / theta_i[i][1] ** 3
    
    return dldo/data.shape[0]

def optimize_step(data, t_i, theta_i, N=3):
    dldt = derivative_t(data, t_i, theta_i, N)
    dldo = derivative_theta(data, t_i, theta_i, N)
    LR = 0.2
    t_i += LR * 2e-1 * dldt
    s = sum(t_i)
    t_i = [t / s for t in t_i]
    theta_i += LR * dldo
    #print(1e-3 * dldt)
    #print(dldo)
    return t_i, theta_i

def optimize(data, t_i, theta_i, N=3):
    losses = []
    for i in range(100):
        loss = dataset_loss(data, t_i, theta_i, N)
        #print("Now(%d): %f" % (i, loss))
        losses.append(loss)
        t_i, theta_i = optimize_step(data, t_i, theta_i, N)
        #print(t_i)
        #print(theta_i)
    return t_i, theta_i, losses

def plot(data, name):
    plt.plot(data)
    plt.savefig(name+".png")
    plt.close()

def do_MLE(data, N, mus=None, sigmas=None, name="raw"):
    # do MLE
    ts = []
    thetas = []
    ls = []
    init_mu = np.array([[1., 1., 1.], [5., 5., 5.], [10., 10., 10.]])
    init_sigma = np.array([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]])

    if mus is not None:
        init_mu = mus
    if sigmas is not None:
        init_sigma = sigmas

    for i in range(3):
        # initialize
        t_i = np.zeros((N,))
        t_i.fill(1./N)
        theta_i = np.zeros((N, 2))
        theta_i[:, 0] = init_mu[:N, i]
        theta_i[:, 1] = init_sigma[:N, i]

        try:
            t_i, theta_i, losses = optimize(data[:, i], t_i, theta_i, N=N)
        except KeyboardInterrupt:
            break
        
        ls.append(np.array(losses))
        ts.append(t_i)
        thetas.append(theta_i)

    mus = thetas
    thetas = np.array(thetas)
    #print(thetas)
    mus = thetas[:, :, 0].transpose()
    sigmas = thetas[:, :, 1].transpose()
    ts = np.array(ts).transpose()

    print("Ts:")
    printable(ts, N)
    print("Mus:")
    printable(mus, N)
    print("sigmas:")
    printable(sigmas, N)

    plt.plot(ls[0])
    plt.plot(ls[1])
    plt.plot(ls[0])
    plt.savefig(name+"N"+str(N)+"MLE.png")
    plt.close()

    return ts, mus, sigmas, thetas

def knn_decision(data, label, mus):
    ml = []
    for x in data:
        y = np.array([np.dot(x - m, x - m) for m in mus])
        ind = y.argmin()
        ml.append(ind)
    ml = np.array(ml)
    count = (ml == label).sum()
    print("acc: %f" % (float(count) / float(label.shape[0])))

def bayes_decision(data, label, ts, thetas):
    ml = []
    for x in data:
        y = np.array([get_fi(x, i, ts[0], thetas[0]) for i in range(3)])
        y *= np.array([get_fi(x, i, ts[1], thetas[1]) for i in range(3)])
        y *= np.array([get_fi(x, i, ts[2], thetas[2]) for i in range(3)])
        ind = y.argmax()
        ml.append(ind)
    ml = np.array(ml)
    count = (ml == label).sum()
    print("acc: %f" % (float(count) / float(label.shape[0])))


# get dataset
params = default_data()
data = generate_data(params, [1000, 600, 1600])

# get test dataset
testdata = generate_data(params, [100, 100, 100])
labels = np.array([0]*100 + [1]*100 + [2]*100)

# show data set
print(data.shape)
scatter3d(data)

# do K-means
N = 3
mus, sigmas = KMeans(data, N)
ts, mus, sigmas, thetas = do_MLE(data, 2, mus, sigmas, name="init")


knn_decision(testdata, labels, mus)
bayes_decision(testdata, labels, ts, thetas)
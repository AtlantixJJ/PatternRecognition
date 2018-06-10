# coding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def printable(data):
    for i in range(data.shape[0]):
        a = "|"
        for j in range(data.shape[1]):
            a += "%.5f|" % data[i][j]

        print(a)


def default_data():
    S1 = np.array([[12, 0],
                [0,  1]])
    S2 = np.array([[8, 3],
                [3, 2]])
    S3 = np.array([[2, 0],
                [0, 2]])
    S = np.array([S1, S2, S3])

    M1 = np.array([1, 1])
    M2 = np.array([7, 7])
    M3 = np.array([15,1])
    M = np.array([M1, M2, M3])
    return M, S

def get_dataset(M, S, n_samples=[1000, 1000, 1000], N=1000):
    """
    create dataset: sample and label
    """
    datas = []
    labels = []
    category = 0
    for m, s, n in zip(M, S, n_samples):
        datas.append(np.random.multivariate_normal(m, s, (n,)))
        labels.append(np.array([category,] * n))
        category += 1

    data = np.concatenate(datas, axis=0)
    label = np.concatenate(labels, axis=0)
    
    # drop to 1000 sample
    rng = np.random.RandomState(1295)
    indice = np.array(range(data.shape[0]))
    rng.shuffle(indice); indice = indice[:N]
    data = data[indice]
    label = label[indice]
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
    c = ['r', 'g', 'b', 'c']
    return [c[v] for v in label]

N = 3
DIM = 2

def lnp_x_mu_sigma(x, mu, sigma, t):
    res = 0

    N = mu.get_shape().as_list()[0]

    for i in range(N):
        m = mu[i:i+1, :]
        s = sigma[i, :, :]

        det = tf.matrix_determinant(s)
        inv = tf.matrix_inverse(s)
        n_sample = tf.cast(tf.shape(x)[0], tf.float64)

        """
        if i == N-1:
            tx = 1 - tf.reduce_sum(t[:-1])
        else:
            tx = t[i]
        a1 =  -0.5 * n_sample * (tf.log((2 * np.pi) ** DIM * det) + 3)
        t_ = tf.log((2 * np.pi) ** DIM * det)
        a2 =  -0.5 * tf.reduce_sum(tf.matmul(x - m, inv) * (x - m))
        """
        a1 = tf.sqrt( (2 * np.pi) ** N * det)
        a2 =  -0.5 * tf.reduce_sum(tf.matmul(x - m, inv) * (x - m), axis=1)

        res += tf.exp(a2) / a1 * t[i]
    res = tf.reduce_sum(tf.log(res))

    return res, det, inv, a1, a2

# draw dataset
M, S = default_data()
dataset, label = get_dataset(M, S)

colors = result2color(label)
plt.scatter(dataset[:, 0], dataset[:, 1], s=1, c=colors)
plt.savefig("dataset_111.png")
plt.close()

def do_MLE(dataset, im=None, isi=None):
    x = tf.placeholder(tf.float64, [None, 2])
    lr = tf.placeholder(tf.float64, [])

    if im is not None:
        init_mu = im
    else:
        init_mu = np.random.uniform(size=(N, DIM))
    
    if isi is not None:
        init_sigma = isi
    else:
        init_sigma = np.random.uniform(size=(N, DIM, DIM))

    # priori
    t = tf.Variable(np.ones((N,), dtype="float64"))
    et = (1+t*t) / tf.reduce_sum(1+t*t)

    # mu
    mu = tf.Variable(init_mu)

    # sigma
    sigma = tf.Variable(init_sigma)
    psigma = tf.matmul(sigma, sigma, transpose_a=True)

    # loss function
    loss, det, inv, a1, a2 = lnp_x_mu_sigma(x, mu, psigma, et)
    loss = -loss

    # Gibbs optimization
    optim_a = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[mu, sigma])
    optim_t = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[t])

    final_t, final_mu, final_sigma = 0, 0, 0

    with tf.Session() as sess:
        # init
        sess.run(tf.global_variables_initializer())
        
        # examine
        #l, d, i, aa1, aa2 = sess.run([loss, det, inv, a1, a2], {x:dataset})
        #print(l, d, i)

        # train
        l = []
        LR = 10
        for i in range(10000):
            loss_, _ = sess.run([loss, optim_a], {x:dataset, lr:LR})
            l.append(loss_)

            # learning rate schedule
            if loss_ > 10000:
                LR = 1
            elif loss_ >= 7000:
                LR = 0.1
            else:
                LR = 0.01
                #loss_, _ = sess.run([loss, optim_t], {x:dataset, lr:0.0001})



            if i % 1000 == 0:
                print("LOSS(%d): %f" % (i, loss_))
                printable(et.eval().reshape(1, -1))
        
        final_t, final_mu, final_sigma = et.eval(), mu.eval(), psigma.eval()

    print("T")
    printable(final_t.reshape(1, -1))
    print("Mu")
    printable(final_mu)
    print("Sigma")
    for i in range(N):
        print(i)
        printable(final_sigma[i])

    plt.plot(l[100:])
    plt.savefig("loss.png")
    plt.close()
    return final_t, final_mu, final_sigma

def estimate(dataset):
    mu = dataset.mean(0, keepdims=True)
    c = dataset.sum(0, keepdims=True)
    XTX = np.dot(dataset.transpose(), dataset)
    t = dataset - mu
    cov = np.dot(t.transpose(), t) / dataset.shape[0]
    #cov = XTX - np.dot(c.transpose(), c)
    return mu, cov

def do_BE(dataset, label):
    mu = []
    cov = []
    for cat in range(N):
        data_cat = dataset[label==cat]
        m, c = estimate(data_cat)
        mu.append(m[0])
        cov.append(c)

    mu = np.array(mu)
    cov = np.array(cov)
    print("Mu")
    printable(mu)
    print("Cov")
    for i in range(N):
        print(i)
        printable(cov[i])

    return mu, cov

# test 2 : BE at 1:1:1
print("test 1 : BE at 1:1:1")
mu0, sigma0 = do_BE(dataset, label)

# test 2 : BE at 0.6:0.3:0.1
print("test 2 : BE at 0.6:0.3:0.1")
mu1, sigma1 = do_BE(dataset, label)

# test 1 : MLE at 1:1:1
print("test 1 : MLE at 1:1:1")
t1, mu1, sigma1 = do_MLE(dataset, mu0, sigma0)

# test 2 : MLE at 0.6, 0.3, 0.1
print("test 2 : MLE at 0.6, 0.3, 0.1")
dataset, label = get_dataset(M, S, [600, 300, 100])
colors = result2color(label)
plt.scatter(dataset[:, 0], dataset[:, 1], s=1, c=colors)
plt.savefig("dataset_631.png")
plt.close()
t2, mu2, sigma2 = do_MLE(dataset, mu1, sigma1)

# test 3 : MLE in 1:1:1 for 300 points
print("test 3 : BE in 1:1:1 for 300 points")
mu1, sigma1 = do_BE(dataset, label)

print("test 3 : MLE in 1:1:1 for 300 points")
dataset, label = get_dataset(M, S, N=300)
t1, mu1, sigma1 = do_MLE(dataset, mu1, sigma1)
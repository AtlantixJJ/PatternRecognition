import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.plot(range(10))
plt.close()

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

seq1 = """AABBCCDD
ABBCBBDD
ACBCBCD
AD
ACBCBABCDD
BABAADDD
BABCDCC
ABDBBCCDD
ABAAACDCCD
ABD""".split("\n")

seq2 = """DDCCBBAA
DDABCBA
CDCDCBABA
DDBBA
DADACBBAA
CDDCCBA
BDDBCAAAA
BBABBDDDCD
DDADDBCAA
DDCAAA""".split("\n")

test_seq = """BADBDCBA
ABBBCDDD
DADBCBAA
CDCBABA
ADBBBCD""".split("\n")

def printable(data):
    for i in range(data.shape[0]):
        a = "|"
        for j in range(data.shape[1]):
            a += "%.5f\t|" % data[i][j]

        print(a)

def printmat(mat):
    print("|" * mat.shape[1])
    a = "|"
    for i in range(mat.shape[1]):
        a += ":--|"
    print(a)
    printable(mat)

def get_trans_mat(N, seqs):
    """
    mat[i, j]: P(j|i)
    """
    mat = np.zeros((N, N), dtype="float32")
    for s in seqs:
        for j in range(1, len(s)):
            c1, c2 = ord(s[j-1]) - ord('A'), ord(s[j]) - ord('A')
            mat[c1, c2] += 1
    #print(mat)
    count = mat.sum(axis=0)
    prior = count.astype("float32") / count.sum()
    mat /= mat.sum(axis=1, keepdims=True)
    #print(mat)
    return mat, prior

def ln_P_seq(seq, mat, prior=None):
    N = mat.shape[0]

    rec_prob, rec_loss = [], []

    if prior is None:
        # uniform prior
        p = 1. / N
        res = -np.log(p)
    else:
        # first ch
        idx = ord(seq[0]) - ord('A')
        p = prior[idx]
        res = -np.log(p)
    
    rec_prob.append(p)
    rec_loss.append(res)

    for i in range(1, len(seq)):
        c1, c2 = ord(seq[i-1]) - ord('A'), ord(seq[i]) - ord('A')

        p = mat[c1, c2]
        if p == 0:
            res += 10000
        else:
            res += -np.log(p)

        rec_prob.append(p)
        rec_loss.append(res)

    return res, rec_prob, rec_loss

def classify(seqs, mats, priors):
    ind_res = []
    for s in seqs:
        res = []
        for j in range(mats.shape[0]):
            l, _, _ = ln_P_seq(s, mats[j], priors[j])
            res.append(l)
        res = np.array(res)

        print(res)
        ind = np.argmax(res) + 1
        ind_res.append(ind)
    return np.array(ind_res).reshape(1, len(seqs))

# build transition matrix
print("Seq1 transition matrix")
mat1, prior1 = get_trans_mat(4, seq1)
printmat(mat1)
printmat(prior1.reshape(1, 4))

print("Seq2 transition matrix")
mat2, prior2 = get_trans_mat(4, seq2)
printmat(mat2)
printmat(prior2.reshape(1, 4))

# classification example
j1, recp1, recl1 = ln_P_seq(test_seq[0], mat1, prior1)
j2, recp2, recl2 = ln_P_seq(test_seq[0], mat2, prior2)

plt.plot(recp1)
plt.plot(recp2)
plt.legend(['w1', 'w2'])
plt.xlabel("No. Character")
plt.ylabel("Transition Probability")
plt.savefig("fig/prob3_1.png")
plt.close()

plt.plot(recl1)
plt.plot(recl2)
plt.legend(['w1', 'w2'])
plt.xlabel("No. Character")
plt.ylabel("Log Probability")
plt.savefig("fig/prob3_2.png")
plt.close()

delta = j1 - j2
first_ch = ord(test_seq[0][0]) - ord('A')
old_prior = np.log(prior2[first_ch])
new_prior = np.exp(-delta + old_prior)
print("New priori: %f" % new_prior)
tmp = prior2[first_ch]
prior2[first_ch] = new_prior
j3, recp3, recl3 = ln_P_seq(test_seq[0], mat2, prior2)
prior2[first_ch] = tmp

plt.plot(recl1)
plt.plot(recl2)
plt.plot(recl3)
plt.legend(['w1', 'w2', 'new_w2'])
plt.xlabel("No. Character")
plt.ylabel("Log Probability")
plt.savefig("fig/prob3_3.png")
plt.close()

# classify all
result = classify(test_seq, np.array([mat1, mat2]), np.array([prior1, prior2]))
printable(result)
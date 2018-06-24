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

test_seq = """ABBBCDDD
DADBCBAA
CDCBABA
ADBBBCD
""".split("\n")

def printable(data):
    for i in range(data.shape[0]):
        a = "|"
        for j in range(data.shape[1]):
            a += "%.5f\t|" % data[i][j]

        print(a)

def printmat(mat):
    print("|" * mat.shape[0])
    a = "|"
    for i in range(mat.shape[0]):
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
    mat /= mat.sum(axis=1, keepdims=True)
    #print(mat)
    return mat

print("Seq1 transition matrix")
printmat(get_trans_mat(4, seq1))
print("Seq2 transition matrix")
printmat(get_trans_mat(4, seq2))
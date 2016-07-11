import numpy as np

def test(M, N, K):
    a = np.zeros((M, K))
    b = np.zeros((K, N))
    c = np.zeros((M, N))

    a += 1
    b += 2
    c += 3
    alpha = 4
    beta = 5
    d = alpha * np.dot(a, b) + beta * c
    print d
    return d[0, 0]

# NN
M = 16
N = 18
K = 17
print "NN", test(M, N, K)

# NT
M = 9
N = 19
K = 13
print "NT", test(M, N, K)

# TN
M = 12
N = 3
K = 8
print "TN", test(M, N, K)

# TT
M = 11
N = 32
K = 9
print "TT", test(M, N, K)

M = 256
N = 2
K = 256
print "BG", test(M, N, K)


import numpy as np

from NASsearch.wl_kernel import WLKernel


class GaussianProcess:
    def __init__(self, kernel_param):
        # self.N = 10  # number of training points.
        # self.n = 50  # number of test points.
        self.s = 0.00005  # noise variance.

        self.X = None
        self.y = None
        self.L = None
        self.kernel_param = kernel_param

    def square_exponential_kernel(self, dist):
        return np.exp(-0.5 * (1 / self.kernel_param) * dist)

    def cal_distance(self, a, b):
        dist = [[] for i in range(len(a))]
        # print(a)
        # print(b)
        # print(len(dist))
        for i in range(len(a)):
            for j in range(len(b)):
                kernel = WLKernel(N=3, net1=a[i], net2=b[j], level=8)
                dist[i].append(kernel.run())
        # print(np.array(dist))
        return np.array(dist)
        # sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
        # return sqdist

    def fit(self, X, y):
        self.X = X
        self.y = np.array(y)
        dist_mat = self.cal_distance(X, X)
        K = self.square_exponential_kernel(dist_mat)
        self.L = np.linalg.cholesky(K + self.s * np.eye(len(X)))

    def predict(self, Xtest):
        dist_mat = self.cal_distance(self.X, Xtest)
        print("dist_mat.shape: ",dist_mat.shape)
        Lk = np.linalg.solve(self.L, self.square_exponential_kernel(dist_mat))
        # print(self.y.shape)
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.y))

        dist_mat = self.cal_distance(Xtest, Xtest)
        K_ = self.square_exponential_kernel(dist_mat)
        s2 = np.diag(K_) - np.sum(Lk ** 2, axis=0)
        s = np.sqrt(s2)

        return mu, s2, s
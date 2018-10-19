import numpy as np
import json

from all_kernel import all_Kernel


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
        l = 9.03240049309828
        sq_f = 2.0169745775312116
        sq_n = 0.22160936170816334

        if len(dist) == len(dist[0]):
            return sq_f * np.exp(-0.5 * (1 / l ** 2) * dist) + np.eye(len(dist)) * sq_n
        else:
            return sq_f * np.exp(-0.5 * (1 / l ** 2) * dist)


    def cal_distance(self, all_nets):
        dist = np.zeros((len(all_nets),len(all_nets)))
        kernel = all_Kernel(N=3, netlist=all_nets, level=8)
        net_vectors = kernel.run()

        print('net_vectors\n',net_vectors)

        for i in range(len(all_nets)):
            for j in range(i+1,len(all_nets)):
                dist[i,j] = np.sum((net_vectors[i]-net_vectors[j])**2)
        for i in range(len(all_nets)):
            for j in range(0,i):
                dist[i,j] = dist[j,i]
        return dist


    def fit_predict(self, X, Xtest):
        self.X = X
        self.y = np.array(y)
        all_dist_mat = self.cal_distance(self.X, X+Xtest)
        self.dist_mat = all_dist_mat[:len(X),:len(X)]
        K = self.square_exponential_kernel(dist_mat)
        self.L = np.linalg.cholesky(K)

        # Lk = np.linalg.solve(self.L, self.square_exponential_kernel(dist_mat))
        Lk = np.linalg.solve(self.L, all_dist_mat[:len(X),len(X):])
        # print(self.y.shape)
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.y))

        # dist_mat = self.cal_distance(Xtest, Xtest)
        K_ = self.square_exponential_kernel(all_dist_mat[len(X):,len(X):])
        s2 = K_ - np.sum(Lk ** 2, axis=0)
        s = np.diag(s2)
        return mu, s2, s

def test():
    archs = []
    no_of_sample = 300
    no_of_line = 1
    with open("test_data/train.txt", "r") as f:
        for line in f.readlines():
            archs.append(json.loads(line.split(" accuracy: ")[0]))
            no_of_line += 1
            if no_of_line > no_of_sample:
                break
    gp = GaussianProcess(80)
    dist_mat = gp.cal_distance(archs)
    counter = 0
    for i in range(dist_mat.shape[0]-1):
        for j in range(i+1, dist_mat.shape[0]):
            if np.array_equal(dist_mat[i,:], dist_mat[j,:]):
                counter += 1
                print('identity: ',i,' and ',j)
    print(counter)

test()

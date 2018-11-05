#! /usr/bin/env python3
import numpy as np
import json
import pickle
from all_kernel import all_Kernel


class GaussianProcess:
    def __init__(self, kernel_param):
        self.s = 0.00005 
        self.X = None
        self.y = None
        self.L = None
        self.kernel_param = kernel_param

    def square_exponential_kernel(self, dist):

        l=31.5917
        sq_f=3.5919
        sq_n=0.4472 # -8.0 score on CV.py

        if len(dist) == len(dist[0]):
            return sq_f * np.exp(-0.5 * (1 / l ** 2) * dist) + np.eye(len(dist)) * sq_n**2
        else:
            return sq_f * np.exp(-0.5 * (1 / l ** 2) * dist)


    def cal_distance(self, all_nets):

        dist = np.zeros((len(all_nets),len(all_nets)))
        net_vectors = self.get_vectors(all_nets)
        for i in range(len(all_nets)):
            for j in range(i+1,len(all_nets)):
                dist[i,j] = np.sum((net_vectors[i]-net_vectors[j])**2)

        for i in range(len(all_nets)):
            for j in range(0,i):
                dist[i,j] = dist[j,i]
        return dist

    def get_vectors(self, all_nets):
        dist = np.zeros((len(all_nets),len(all_nets)))
        kernel = all_Kernel(N=3, netlist=all_nets, level=8)
        net_vectors = kernel.run()
        # reweight net_vectors
        print('vector shape:',net_vectors.shape)
        with open('../reweightForWL/weight.pkl','rb') as f:
            weight = pickle.load(f)
        weight = np.tile(weight.data.numpy(),(len(all_nets),1))
        print('weight shape:',weight.shape)
        net_vectors = np.multiply(net_vectors,weight)
        print('net_vectors\n',net_vectors)

        return net_vectors


    def fit_predict(self, X, Xtest, y):
        self.X = X
        self.y = np.array(y)
        all_dist_mat = self.cal_distance(X+Xtest)
        self.dist_mat = all_dist_mat[:len(X),:len(X)]
        K = self.square_exponential_kernel(self.dist_mat)
        self.L = np.linalg.cholesky(K)
        Lk = np.linalg.solve(self.L, self.square_exponential_kernel(all_dist_mat[:len(X),len(X):]))
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.y))
        K_ = self.square_exponential_kernel(all_dist_mat[len(X):,len(X):])
        s2 = K_ - np.sum(Lk ** 2, axis=0)
        s = np.diag(s2)
        return mu, s2, s

def test():
    archs = []
    no_of_sample = 136+256+256+256
    no_of_line = 1
    label = []


    with open("test_data/stage2.txt", "r") as f:
        for line in f.readlines():
            archs.append(json.loads(line.split(" accuracy: ")[0]))
            label.append(float(line.split(" accuracy: ")[1][:-1]))
            no_of_line += 1
            if no_of_line > no_of_sample:
                break
    

    label = np.array(label)
    label = (label-label.mean())/label.std()
    gp = GaussianProcess(80)
    

    dist_mat = gp.cal_distance(archs)
    print(dist_mat)
    with open('dist_mat.pkl','wb') as opf:
        pickle.dump(dist_mat,opf)
    counter = 0
    for i in range(dist_mat.shape[0]-1):
        for j in range(i+1, dist_mat.shape[0]):
            if np.array_equal(dist_mat[i,:], dist_mat[j,:]):
                counter += 1
                print('identity: ',i,' and ',j)
    print(counter)
    
test()

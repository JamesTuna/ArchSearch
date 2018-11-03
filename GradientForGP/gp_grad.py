#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as pl
import torch
from torch.autograd import Variable
#from gaussian_process.wl_kernel import WLKernel
#from NASsearch.wl_kernel import WLKernel
import pickle,scipy,math

def isPSD(A, tol=1e-8):
    E = np.linalg.eigvalsh(A.data.numpy())
    return np.all(E > -tol)

class GaussianProcess:
    def __init__(self, sq_f, l, sq_n, lr):
        self.sq_n = Variable(torch.Tensor([sq_n]), requires_grad=True) # noise variance.
        self.l = Variable(torch.Tensor([l]), requires_grad=True)
        self.sq_f = Variable(torch.Tensor([sq_f]), requires_grad=True)
        self.X = None
        self.y = None
        self.L = None
        self.dist_mat = None
        self.grads = None
        self.lr = lr
        self.lr_decay = 200
        self.optimizer = torch.optim.Adam([self.sq_f,self.l],lr=lr)

    def cal_distance(self, a, b):
        dist = [[] for i in range(len(a))]
        # print(len(dist))
        for i in range(len(a)):
            for j in range(len(b)):
                kernel = WLKernel(3, a[i], b[j], 20)
                dist[i].append(kernel.run())
        # print(np.array(dist))
        return np.array(dist)
        # sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
        # return sqdist

    def square_exponential_kernel(self, dist):
        if len(dist) == len(dist[0]):
            return self.sq_f*np.exp(-0.5 * (1 / self.l**2) * dist)+np.eye(len(dist))*self.sq_n**2
        else:
            return self.sq_f*np.exp(-0.5 * (1 / self.l**2) * dist)

    def fit(self, X, y):
        self.X = X
        self.y = np.array(y)
        dist_mat = self.cal_distance(X, X)
        K = self.square_exponential_kernel(dist_mat)
        self.dist_mat = dist_mat
        print(dist_mat)
        #self.L = np.linalg.cholesky(K)

    def learn(self,iter):
        # cashed value: dist_mat, X, y
        lr = self.lr
        for i in range(iter):
            # y y_mean column vector with shape (n,1)
            sqr_dist_mat = torch.Tensor(self.dist_mat).detach()
            y = torch.Tensor(np.reshape(self.y,(len(self.y),1))).detach()
            y_mean = torch.Tensor(np.zeros((len(self.y),1))).detach()
            I = torch.Tensor(np.eye(len(self.y))).detach()
            cov = self.sq_f*torch.exp(-1/(2*self.l**2)*sqr_dist_mat)+I*self.sq_n**2
            if not isPSD(cov):
                print("cov is not PSD")
            # calculate determinent of cov
            det = cov.det()
            #log marginal likelihood, ignore term -(n/2)log(1/2π)
            log_like = -(-0.5*det.log()-0.5*(y-y_mean).transpose(0,1).mm((cov.inverse())).mm((y-y_mean)))
            self.optimizer.zero_grad()
            log_like.backward()
            if i >= 0:
                print("iter: ",i)
                print("l: ",self.l," sq_f: ",self.sq_f," sq_n: ",self.sq_n)
                print("det of K:\n",det.data.numpy())
                print("loglike:\n",-log_like.data.numpy())
                print("grads: ",[-self.l.grad.data.numpy()[0],-self.sq_f.grad.data.numpy()[0],-self.sq_n.grad.data.numpy()[0]])
                print("")
            if torch.isnan(log_like):
                print("grad nan at iter: ",i)
                print("det: ",det.data.numpy())
                break
            '''
            # manually update parameter
            self.l += 100*lr*l.grad.data.numpy()[0]
            self.sq_f += 100*lr*sq_f.grad.data.numpy()[0]
            self.sq_n += lr*sq_n.grad.data.numpy()[0]
            '''
            self.optimizer.step()


    def predict(self, Xtest):
        dist_mat = self.cal_distance(self.X, Xtest)
        Lk = np.linalg.solve(self.L, self.square_exponential_kernel(dist_mat))
        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.y))
        dist_mat = self.cal_distance(Xtest, Xtest)
        K_ = self.square_exponential_kernel(dist_mat)
        s2 = np.diag(K_) - np.sum(Lk ** 2, axis=0)
        s = np.sqrt(s2)
        return mu, s2, s

net_list = []
acc_list = []
i = 0
inputF = open('../Kernel_optimization/dist_mat.pkl','rb')
with open("../Kernel_optimization/test_data/stage4.txt") as file:
    for line in file:
        i += 1
        net,acc = line.split(" accuracy: ")
        net_list.append(net)
        acc_list.append(float(acc[:-1]))
        if i == 136+256+256+256:
            break

y = np.array(acc_list)
y = (y - y.mean())/y.std()
print(y)

# normalized data for first 136 samples
print("y len: ",len(y))
# sq_f,l,sq_n,lr
model = GaussianProcess(1.221,66.365,math.sqrt(0.63),0.05)
model = GaussianProcess(9.29,195,math.sqrt(0.63),0.05)
model = GaussianProcess(0.5,70,math.sqrt(0.85),0.05)
print("now fitting model...")
model.dist_mat = np.array(pickle.load(inputF))[:136+256+256+256,:136+256+256+256]
print(model.dist_mat)
print(model.dist_mat.shape)
model.y = y
inputF.close()
print("now learning hp...")
model.learn(20000)
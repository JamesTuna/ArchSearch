import numpy as np
import matplotlib.pyplot as pl
import torch
from torch.autograd import Variable
#from gaussian_process.wl_kernel import WLKernel
from NASsearch.wl_kernel import WLKernel
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
inputF = open('dist_mat.txt','rb')
inputF = open('../Kernel_optimization/dist_mat.pkl','rb')
with open("train_gp.txt") as file:
    for line in file:
        i += 1
        net,acc = line.split(" accuracy: ")
        net_list.append(net)
        acc_list.append(float(acc[:-1]))
        if i == 136:
            break
X = net_list[:]
y = acc_list[:]
# normalized data for first 136 samples
y = [-2.2190403290819622, -1.6994628448868245, -2.132444265237555, -2.649135140562633, -1.546475878159735, -1.3530774883805015, -2.40233586309816, -1.9996630100365373, -0.7815421456529805, -1.194317854477639, -1.6171964190653334, -0.2504182259776035, 0.37163139556509733, 0.27637605567485724, -1.3530774883805015, -0.573710711523448, 1.123575663438756, 0.813273193090354, 0.5563699587341513, -0.2576350234349264, 0.5072992596361747, 0.7699746106038035, 0.13060533584074382, -0.5434021442343401, -0.03537109551944155, -0.052689647611106914, -0.7526771580810759, 0.6487403414473718, 1.1307924608960789, 1.107700691064294, 0.2099857033565218, -1.7875024884484352, -1.1062782109160283, -2.018425692409753, -2.44996353304328, -1.47142569923122, -0.9244257060526874, -2.523570682818938, -0.2316560941687348, 0.7540985371006479, 0.39905390454849216, -0.30093250479278333, -0.3673228571111188, 1.0499696147917912, 0.8002831778929114, -0.28650001100683103, -0.18114071422486147, 0.713687114048504, 0.826262107159103, 0.4250328338146837, -0.5491753619744597, -0.18258429394206474, -0.21722249925408899, -0.8118507129420884, 0.3153438990097978, 1.3934678118637076, 1.2419238742894745, -0.22155323840569885, -2.5351165677348306, -1.7182249766956932, -1.9433755134812378, 0.357198901779145, 0.7093574760255876, 0.3658581778249777, 0.0772038975911574, 0.5477106826883187, 0.9215185481780366, 0.803170337327318, 0.6848221264765993, 0.72378996981154, 0.9994553359766113, 1.2130588867175698, 1.0586288908376238, -0.42072429536070516, -0.05557680704551349, 0.09452355081151628, -0.8883439210234598, 1.0514120933803008, 1.3732621003376355, 1.2087292486946535, 0.010813545272821957, -0.10031786812057371, 0.5823488880003429, 0.29658176720092916, -0.2980464464870703, 0.9301778242238692, 0.6097713969837377, 0.6660588935390372, -0.8132942926592918, 0.47121747460694724, 0.3023549849410488, 0.4899796064158159, 0.22008855911955777, 0.04400927199633644, 0.06132892521669532, -0.10464750614349004, 0.40627070200581505, 1.0326499615714322, 1.0456399767688747, 0.37451855499950387, -0.7598928544097053, 0.793067481564282, 1.0182174677854798, -1.4584362345981243, 0.2792621139805703, 0.3730749752823006, 1.0860513998210186, -0.2547478640005198, 0.5664739156258808, 0.5592571181685579, 0.30668462296396515, -0.02093860173348924, 1.2404813957009648, 1.2462546134410843, 1.0066710323052406, -1.3256555299614534, 0.5404949863596893, 1.029763903265719, -0.45392002208421967, -0.007948586536046695, 0.43657926929492297, 0.3802917727396235, -0.2027900054681367, 0.9965681765422046, 1.0975967341725643, 1.1986263929316177, -1.0341146408575732, -0.25186180569480676, -0.676182848871011, 0.24750996697425912, 1.185636377734175, 0.7728606689095165, 0.2864789114378932, 0.6646153138218339, 1.2909956745161446, 0.4668878365840309]
print("X len: ",len(X))
print("y len: ",len(y))
# sq_f,l,sq_n,lr
model = GaussianProcess(3.0,12.0,math.sqrt(0.2),0.01)
print("now fitting model...")
#model.fit(X,y)
#pickle.dump(model.dist_mat,output)
model.X = X
model.y = y
model.dist_mat = np.array(pickle.load(inputF))[:136,:136]
inputF.close()
print("now learning hp...")
model.learn(20000)
#! /usr/bin/env python3
import matplotlib.pyplot as plt
import pickle,torch
import numpy as np
from torch.autograd import Variable
with open('./data.pkl','rb') as f:
	net_v,label = pickle.load(f)

STEPS = 100000
K = 0.1
LR = 0.01
N = net_v.shape[0]
dim = net_v.shape[1]
SAMPLES = np.arange(int(N*(N-1)/2))
# square difference of xs
# abs difference of ys
# square weight
sqd_x = np.array([np.zeros(dim) for i in range(int(N*(N-1)/2))])
d_y = np.zeros((int(N*(N-1)/2),1))
counter = 0
for i in range(N-1):
	for j in range(i+1,N):
		sqd_x[counter] = ((net_v[i]-net_v[j])**2)
		d_y[counter] = (np.abs(label[i]-label[j]))
		counter +=1
'''
# detect those points that are outliers (namely large ratio of d_y/sqd_x)
ratios = d_y.reshape(-1)/np.sum(sqd_x,1)
r_mean = np.mean(ratios)
outliers = ratios>2*r_mean
number_outliers = len(ratios[outliers])
print(outliers,number_outliers)
SAMPLES = SAMPLES[outliers]
'''

print(sqd_x)
sq_weight = Variable(torch.Tensor(np.ones(dim)),requires_grad=True)
#sq_weight = sq_weight/torch.sum(sq_weight) # normalize squared weight, same meaning as keeping the norm of weight 1
sqd_x = torch.Tensor(sqd_x)
d_y = torch.Tensor(d_y)
optimizer = torch.optim.Adam([sq_weight],lr=LR)


def plot(sqd_x,weight,d_y):
	x = torch.sum(sqd_x * weight,1).data.numpy().reshape(-1)
	print(x)
	x = np.sqrt(x)
	plt.scatter(x,d_y.data.numpy().reshape(-1))
	plt.plot(x,K*np.array(x),'r-')
	plt.ion()
	plt.pause(5)
	plt.close()



def batch_opt(batch_indices):
	global K, sq_weight, optimizer, sqd_x, d_y
	optimizer.zero_grad()
	# normalize weight
	#sq_weight = sq_weight/torch.sum(sq_weight)
	batch_x = sqd_x[batch_indices]
	batch_y = d_y[batch_indices].view(-1)
	bias = batch_y - torch.Tensor([K]) * torch.sqrt(torch.sum(batch_x * sq_weight,1))
	# clip loss
	loss = torch.sum(torch.max(torch.Tensor(np.zeros(len(batch_indices))), bias))
	print('loss: ',loss)
	loss.backward()
	optimizer.step()


for i in range(STEPS):
	np.random.shuffle(SAMPLES)
	sample_indices = SAMPLES[:1000]
	#print(sample_indices)
	loss = batch_opt(sample_indices)
	print(i,sq_weight)
	if i%200 == 0:
		# refit K
		sum_dist = torch.sum(torch.sqrt(torch.sum(sqd_x * sq_weight,1))).data.numpy()
		sum_label_diff = torch.sum(d_y).data.numpy()
		K = 3*sum_label_diff/sum_dist
		print('K: ',K)
	if i%500 == 0:
		plot(sqd_x,sq_weight,d_y)


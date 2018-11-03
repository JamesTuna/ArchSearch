#! /usr/bin/env python3
import matplotlib.pyplot as plt
import pickle,torch
import numpy as np
from torch.autograd import Variable
with open('./data.pkl','rb') as f:
	net_v,label = pickle.load(f)

STEPS = 10000
K = 0.001
LR = 0.01
N = net_v.shape[0]
SAMPLES = np.arange(N)
print(SAMPLES)


label = torch.Tensor(label)
dim = net_v.shape[1]
weight = Variable(torch.Tensor(np.random.randn(dim)),requires_grad=True)
weight = Variable(torch.Tensor(np.ones(dim)),requires_grad=True)
net_v = torch.Tensor(net_v).detach()
transformed_v = net_v * weight

def plot(transformed_v,label):
	x =[]
	y =[]
	for i in range(0,len(label)-1):
		for j in range(i+1,len(label)):
			x.append(np.sum((transformed_v[i]-transformed_v[j])**2))
			y.append(np.abs(label[i]-label[j]))
	plt.scatter(x,y)
	plt.plot(x,K*np.array(x),'r-')
	plt.show()



def batch_opt(batch_v, batch_label,optimizer):
	global K
	optimizer.zero_grad()
	loss = torch.Tensor([0])
	sample_dist = 0
	sample_label_dist = 0
	for i in range(batch_v.shape[0]-1):
		for j in range(i+1,batch_v.shape[0]):
			dist = torch.sum((batch_v[i]-batch_v[j])**2)
			label_dist = torch.abs(batch_label[i]-batch_label[j])
			a = max(0,label_dist - torch.Tensor([K]) * dist)
			sample_dist += dist.data.numpy()
			sample_label_dist += label_dist.data.numpy()
			loss += a + 0.0001*torch.sum(weight**2)
	print('loss:',loss)
	if loss > 0:
		loss.backward()
		optimizer.step()
	K = 5*(sample_label_dist/sample_dist)
	print('change K to', K)
	return loss

'''
def triple_opt(triple_v, labels, optimizer):
	actual_d12 = torch.abs(batch_label[0]-batch_label[1])
	actual_d13 = torch.abs(batch_label[0]-batch_label[2])
	# L((xi, x+i , x−i ); w)=max(0,1 − Sw(xi, x+i )+Sw(xi, x−i ))
	if actual_d13 > actual_d12:
		loss = 
'''

optimizer = torch.optim.Adam([weight],lr=LR)

for i in range(STEPS):
	transformed_v = net_v * weight
	np.random.shuffle(SAMPLES)
	samples = SAMPLES[:30]
	#print(samples)
	loss = batch_opt(transformed_v[samples],label[samples],optimizer)
	print(i,weight)
	if i%500 == 0:
		plot(transformed_v.data.numpy(),label.data.numpy())

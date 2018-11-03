#! /usr/bin/env python3
from all_gp import GaussianProcess
import numpy as np
import math,json

class CV(object):
	def __init__(self,sample_list,n_fold,y):
		self.data = sample_list
		self.n_fold = n_fold
		self.N = len(sample_list)
		self.N_test = math.ceil(self.N/n_fold)
		self.N_train = self.N - self.N_test
		self.y = y
		self.gp = GaussianProcess(80)
		# for reuse
		self.all_dist_mat = self.gp.cal_distance(self.data) # l2_norm ** 2
		self.all_dist_mat = self.gp.square_exponential_kernel(self.all_dist_mat)


	def split(self):
		shuffled = np.arange(self.N)
		np.random.shuffle(shuffled)
		# reordering row and column in all_dist_mat
		self.ordered_dist_mat = (self.all_dist_mat[shuffled,:])[:,shuffled]
		self.y_train = (self.y[shuffled])[:self.N_train]
		self.y_test = (self.y[shuffled])[self.N_train:]

	def fit_predict(self):
		K = self.ordered_dist_mat[:self.N_train,:self.N_train]
		#print(K.shape)
		L = np.linalg.cholesky(K)
		#print(L.shape)
		#print(self.ordered_dist_mat[:self.N_train,self.N_train:].shape)
		Lk = np.linalg.solve(L, self.ordered_dist_mat[:self.N_train,self.N_train:])
		K_ = self.ordered_dist_mat[self.N_train:,self.N_train:]
		s2 = K_ - np.sum(Lk ** 2, axis=0)
		self.test_mean = np.dot(Lk.T, np.linalg.solve(L, self.y_train))
		self.test_std = np.sqrt(np.diag(s2))

	def give_score(self):
		# calculate log likelihood of data in validation set
		cov = self.ordered_dist_mat[self.N_train:,self.N_train:]
		det = np.linalg.det(cov)
		deviation = self.y_test - self.test_mean
		log_like = - np.log(det) - (deviation.T).dot(np.linalg.inv(cov)).dot(deviation)
		return log_like

	def test(self):
		scores = []
		for i in range(100*self.n_fold):
			self.split()
			self.fit_predict()
			score = self.give_score()
			scores.append(score)
		scores = np.array(scores)
		print(scores.mean())


archs = []
no_of_sample = 392
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
tester = CV(archs,10,label)
tester.test()


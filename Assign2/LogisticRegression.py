import numpy as np
import pandas as pd
import random
import sys

import Preprocessing

def Sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res, 1e-6, 1-1e-6)

def Adagrad(train, label):
	t = 0
	epoch = 50000
	alpha = 0.01
	epsilon = 1e-8

	acc    = np.full(train.shape[1], 0.0)
	weight = np.full(train.shape[1], 1.0)	# random

	while t <= epoch:
		t += 1

		# compute loss function and regularized gradient
		temp = np.dot(train, weight)
		pred = Sigmoid(temp)
		loss = label - pred
		grad = -2 * np.dot(train.transpose(), loss) + 2 * 0.001 * np.sum(weight**2)

		acc += grad ** 2
		weight -= alpha * grad / (np.sqrt(acc) + epsilon)

		# printerror = (np.sum((np.dot(train, weight) - label) ** 2) / train.shape[0])**0.5
		# if t % 1000 == 0: print("iteration: %d" % t, "error: %f" % printerror)
	return weight

if __name__ == "__main__":

	if len(sys.argv) != 6:
		print ("Usage: python script <normalization> <train file> <label file> <test file> <output file>")
		exit(0)

	normalization = sys.argv[1]
	train, label, test = Preprocessing.LoadData(sys.argv[2], sys.argv[3], sys.argv[4])
	
#	if normalization == "standardize": train, test = Preprocessing.Standardize(train, test)
#	if normalization == "rescale":     train, test = Preprocessing.Rescale(train, test)

	train = np.concatenate((train, (train[:,0]**2).reshape(-1,1), (train[:,0]**3).reshape(-1,1)), axis = 1)
	train = np.concatenate((train, (train[:,1]**2).reshape(-1,1), (train[:,1]**3).reshape(-1,1)), axis = 1)
	train = np.concatenate((train, (train[:,3]**2).reshape(-1,1), (train[:,3]**3).reshape(-1,1)), axis = 1)
	train = np.concatenate((train, (train[:,4]**2).reshape(-1,1), (train[:,4]**3).reshape(-1,1)), axis = 1)
	train = np.concatenate((train, (train[:,5]**2).reshape(-1,1), (train[:,5]**3).reshape(-1,1)), axis = 1)
	train = np.concatenate((np.ones((train.shape[0], 1)), train), axis = 1)

	test  = np.concatenate((test,  (test[:,0] **2).reshape(-1,1), (test[:,0] **3).reshape(-1,1)), axis = 1)
	test  = np.concatenate((test,  (test[:,1] **2).reshape(-1,1), (test[:,1] **3).reshape(-1,1)), axis = 1)
	test  = np.concatenate((test,  (test[:,3] **2).reshape(-1,1), (test[:,3] **3).reshape(-1,1)), axis = 1)
	test  = np.concatenate((test,  (test[:,4] **2).reshape(-1,1), (test[:,4] **3).reshape(-1,1)), axis = 1)
	test  = np.concatenate((test,  (test[:,5] **2).reshape(-1,1), (test[:,5] **3).reshape(-1,1)), axis = 1)
	test  = np.concatenate((np.ones((test.shape[0],  1)), test ), axis = 1)

	# adopt logistic regression model
	weight = Adagrad(train, label)

	# write the results to file
	index = np.arange(1, 16282)
	index = index.astype("int")

	predict = np.dot(test, weight)
	for k in range(len(predict)):
		if predict[k] >= 0: predict[k] = 1
		if predict[k] <= 0: predict[k] = 0
	predict = predict.astype("int")

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(sys.argv[5], header = ["id", "label"], index = None)

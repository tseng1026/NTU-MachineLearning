import numpy as np
import pandas as pd
import random
import sys

import Preprocessing

dim = 106

def Sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res, 1e-6, 1-1e-6)

def BayesRule(train, label):
	num = train.shape[0]

	# compute the average value
	cnt = np.zeros((2, 1))
	avg = np.zeros((2, dim))
	for i in range(num):
		cnt[label[i]] += 1
		avg[label[i]] += train[i]
	avg /= cnt

	# compute the sigma value
	sig = np.zeros((2, dim, dim))
	for i in range(num):
		sig[label[i]] += np.dot(np.transpose([train[i] - avg[label[i]]]), [(train[i] - avg[label[i]])])
	sig[0] /= cnt[0]
	sig[1] /= cnt[1]
	share = cnt[0] / num * sig[0] + cnt[1] / num * sig[1]

	inv = np.linalg.inv(share)
	weight = np.dot((avg[1] - avg[0]), inv)
	biased = 0
	biased += 0.5 * np.dot(np.dot(avg[0].T, inv), avg[0])
	biased -= 0.5 * np.dot(np.dot(avg[1].T, inv), avg[1])
	biased += np.log(float(cnt[1])/cnt[0])
	return weight, biased

if __name__ == "__main__":

	if len(sys.argv) != 6:
		print ("Usage: python script <normalization> <train file> <label file> <test file> <output file>")
		exit(0)

	normalization = sys.argv[1]
	train, label, test = Preprocessing.LoadData(sys.argv[2], sys.argv[3], sys.argv[4])
	
	if normalization == "standardize": train, test = Preprocessing.Standardize(train, test)
	if normalization == "rescale":     train, test = Preprocessing.Rescale(train, test)

	# adopt probabilistic generative model
	weight, biased = BayesRule(train, label)

	# write the results to file
	index = np.arange(1, 16282)
	index = index.astype("int")

	temp = np.dot(test, weight) + biased
	predict = Sigmoid(temp)
	predict = np.around(predict)
	predict = predict.astype("int")
	
	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(sys.argv[5], header = ["id", "label"], index = None)
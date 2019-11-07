import numpy as np
import pandas as pd
import sys

import Preprocessing

def Adam(train, label):
	# set the parameters
	t = 0
	batch = 64
	epoch = 5000
	alpha = 0.01
	beta1 = 0.9
	beta2 = 0.99
	epsilon = 1e-8
	batch_size = 64

	mom    = np.full(train.shape[1], 0)		# momentum
	vel    = np.full(train.shape[1], 0)		# velocity
	weight = np.full(train.shape[1], 0.1)	# weight

	while t <= epoch:
		# for b in range(train.shape[0] // batch):
		# 	t += 1
		# 	trainTmp = train[b * batch: (b + 1) * batch]
		# 	labelTmp = label[b * batch: (b + 1) * batch]
		t += 1
		trainTmp = train
		labelTmp = label

		# compute loss function and regularized gradient
		loss = labelTmp - np.dot(trainTmp, weight)
		grad = -2 * np.dot(trainTmp.transpose(), loss) + 2 * 0.001 * np.sum(weight**2)

		mom = beta1 * mom + (1-beta1) * grad
		vel = beta2 * vel + (1-beta2) * grad**2
		momCor = mom / (1 - beta1**t)
		velCor = vel / (1 - beta2**t)
		weight -= alpha * momCor / (np.sqrt(velCor) + epsilon)

		printerror = (np.sum((np.dot(trainTmp, weight) - labelTmp) ** 2) / trainTmp.shape[0])**0.5
		if t % 1000 == 0: print("iteration: %d" % t, "error: %f" % printerror)
	return weight

def Adagrad(train, label):
	t = 0
	batch = 64
	epoch = 10000
	alpha = 0.01
	epsilon = 1e-8

	acc    = np.full(train.shape[1], 0.0)
	weight = np.full(train.shape[1], 0.1)

	while t <= epoch:
		# for b in range(train.shape[0] // batch):
		# 	t += 1
		# 	trainTmp = train[b * batch: (b + 1) * batch]
		# 	labelTmp = label[b * batch: (b + 1) * batch]
		t += 1
		trainTmp = train
		labelTmp = label

		# compute loss function and regularized gradient
		loss = labelTmp - np.dot(trainTmp, weight)
		grad = -2 * np.dot(trainTmp.transpose(), loss) + 2 * 0.001 * np.sum(weight**2)

		acc += grad ** 2
		weight -= alpha * grad / (np.sqrt(acc) + epsilon)

		printerror = (np.sum((np.dot(trainTmp, weight) - labelTmp) ** 2) / trainTmp.shape[0])**0.5
		if t % 1000 == 0: print("iteration: %d" % t, "error: %f" % printerror)
	return weight

if __name__ == "__main__":

	if len(sys.argv) != 4:
		print ("Usage: python script <algorithm> <output file>")
		exit(0)

	algorithm = sys.argv[1]

	# load and preprocess
	data1 = Preprocessing.LoadData("year1_data.csv", 1)
	data2 = Preprocessing.LoadData("year2_data.csv", 1)
	test  = Preprocessing.LoadData("testing_data.csv", 0)
	
	data  = np.concatenate((data1, data2), axis = 1)

	data = Preprocessing.Extraction(data, 1)
	test = Preprocessing.Extraction(test, 0)
	# data = Preprocessing.Recovery(data)
	
	lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	data = Preprocessing.Deletion(data, lst, 1)
	test = Preprocessing.Deletion(test, lst, 0)
	data = np.concatenate((np.ones((data.shape[0], 1)), data), axis = 1)
	test = np.concatenate((np.ones((test.shape[0], 1)), test), axis = 1)
	np.save("data", data)
	np.save("test", test)

	# separate the data of feature vector and the realistic result
	train = data[:, : data.shape[1] - 1]
	label = data[:,   data.shape[1] - 1]
	np.save("train", train)
	np.save("label", label)

	lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
	data = np.load("data.npy")
	test = np.load("test.npy")
	train  = np.load("train.npy")
	label  = np.load("label.npy")
	weight = np.full(train.shape[1], 1)

	# shuffle the training data
	k = np.arange(train.shape[0])
	np.random.shuffle(k)
	train = train[k]
	label = label[k]

	# execute each algorithm
	if algorithm == "adam":    weight = Adam   (train, label)
	if algorithm == "adagrad": weight = Adagrad(train, label)
	np.save("weight", weight)

	# write the results to file
	index = np.arange(500)
	index = index.astype("str")
	index = ["id_" + content for content in index]
	index = np.array(index)

	predict = np.dot(test, weight)
	for k in range(len(predict)):
		if predict[k] <=  2: predict[k] = test[k][test.shape[1] - len(lst) + lst.index(9)]
		if predict[k] > 100: predict[k] = test[k][test.shape[1] - len(lst) + lst.index(9)]

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(sys.argv[2], header = ["id", "value"], index = None)
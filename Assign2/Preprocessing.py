import numpy as np
import pandas as pd

def LoadData(filename_train, filename_label, filename_test):
	# read in the data
	train = pd.read_csv(filename_train)
	label = pd.read_csv(filename_label, header = None)
	test  = pd.read_csv(filename_test)

	train = train.values
	label = label.values.reshape(-1)
	test  = test.values

	return train, label, test

def Standardize(data, test):
	# consider the data both in training and testing
	totl = np.concatenate((data, test), axis = 0)
	mean = np.mean(totl, axis = 0)
	strd = np.std (totl, axis = 0)

	# only normalize the specific columns
	index = [0, 1, 3, 4, 5]
	meanVec = np.zeros(totl.shape[1])
	strdVec = np.ones (totl.shape[1])
	meanVec[index] = mean[index]
	strdVec[index] = strd[index]

	norm = (totl - meanVec) / strdVec
	dataNorm = norm[0:data.shape[0]]
	testNorm = norm[data.shape[0]:]

	return dataNorm, testNorm

def Rescale(data, test):
	# consider the data both in training and testing
	totl = np.concatenate((data, test), axis = 0)
	lrge = np.max(totl, axis = 0)
	smll = np.min(totl, axis = 0)

	# only rescale the specific columns
	index = [0, 1, 3, 4, 5]
	smllVec = np.zeros(totl.shape[1])
	lrgeVec = np.ones (totl.shape[1])
	smllVec[index] = smll[index]
	lrgeVec[index] = lrge[index]

	scle = (totl - smllVec) / (lrgeVec - smllVec)
	dataScle = scle[0:data.shape[0]]
	testScle = scle[data.shape[0]:]

	return dataScle, testScle
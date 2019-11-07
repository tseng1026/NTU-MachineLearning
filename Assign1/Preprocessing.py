import numpy as np
import pandas as pd

def LoadData(filename, test_or_train):
	# read in the data
	load = pd.read_csv(filename)
	load = load.replace("NR", "0")
	load = load.fillna("0")
	if test_or_train == 1: load = load.drop(columns = ["日期"])
	if test_or_train == 0: load = load.drop(columns = ["id"])
	
	data = []
	year = len(load) // 18
	for labl in range(18):
		lab = load.iloc[labl]["測項"]
		now = load[load["測項"] == lab]
		now = now.drop(columns = ["測項"])
		
		# remove the meaningless characters
		for date in range(year):
			tmp = now.iloc[date]
			tmp = tmp.replace("NR", "0")
			tmp = tmp.str.strip("#")
			tmp = tmp.str.strip("*")
			tmp = tmp.str.strip("x")
			tmp = tmp.str.strip("A")
			data.append(tmp)
	
	data = np.array(data).reshape(18, year, load.shape[1] - 1).astype("float")
	print ("[Done] Loading data from file %s!" % filename)
	return data

def Extraction(data, test_or_train):
	# extract the features of the data
	year = data.shape[1]

	if test_or_train == 0:
		res = np.swapaxes(data, 0, 1)
		res = np.reshape (res, (year, 18 * 9))
		print ("[Done] Extracting features from testing data!")
		return res

	if test_or_train == 1:
		data = np.reshape(data, (18, year * 24))

		res = []
		for k in range(year * 24 - 9):
			tmp = np.zeros((18 * 9 + 1))
			tmp[: 18 * 9] = np.reshape(data[:, k : k + 9], (-1, 18 * 9))
			tmp[  18 * 9] = data[9, k + 9]
			if Validation(tmp): res.append(tmp)

		res = np.array(res)
		print ("[Done] Extracting features from training data!")
		return res

def Validation(data):
	if data[18 * 9] <=  2: return False
	if data[18 * 9] > 100: return False

	for k in range(9):
		if data[9 * 9 + k] <=  2: return False
		if data[9 * 9 + k] > 100: return False
	return True

def Recovery(data):
	hour = data.shape[1]
	for lab in range(18):
		
		pre = 0
		cnt = 0
		for k in range(hour):
			if data[lab][k] != -1:
				for now in range(1, cnt+1):
					gap = (data[lab][k] - pre) / (cnt + 1)
					data[lab][k - now] = data[lab][k] - gap * now
				pre = data[lab][k]
				cnt = 0
			
			if data[lab][k] == -1:
				cnt = cnt + 1

	return data

def Deletion(data, lst, test_or_train):
	res = []
	for num in range(data.shape[0]):
		tmp = []

		for fit in range(data.shape[1] - test_or_train):
			if fit % 18 in lst: tmp.append(data[num][fit])
		if test_or_train == 1: tmp.append(data[num][data.shape[1] - 1])
		res.append(tmp)
	
	res = np.array(res)
	print ("[Done] Delete unnecessary features!")
	return res
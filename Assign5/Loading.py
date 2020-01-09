import numpy as np
import pandas as pd
import torch
from   torch.utils.data import Dataset

import Preprocessing
import Vectorizing

def LoadTrn(com, lab, mode="rnn"):
	import random
	com = pd.read_csv(com)
	lab = pd.read_csv(lab)
	com = com[["comment"]].comment.tolist()
	lab = lab[["label"]].label.tolist()
	
	com = Preprocessing.Tokenize(com)
	com = Preprocessing.Lematize(com)
	com = Preprocessing.Garbages(com)
	com = Preprocessing.Stopword(com)
	if mode == "rnn": com = Vectorizing.Vectorize(com)
	if mode == "bow": com = Vectorizing.BagOfWord(com)
	
	com  = np.array(com)
	data = list(zip(com, lab))
	np.save("data.npy", data)

	data = np.load("data.npy", allow_pickle=True)
	data = list(data)
	random.shuffle(data)
	data = np.array(data)

	train = data[:10000]
	valid = data[10000:]
	return train, valid

def LoadTst(com, mode="rnn"):
	com = pd.read_csv(com)
	com = com[["comment"]].comment.tolist()

	com = Preprocessing.Tokenize(com)
	com = Preprocessing.Lematize(com)
	com = Preprocessing.Garbages(com)
	com = Preprocessing.Stopword(com)
	if mode == "rnn": com = Vectorizing.Vectorize(com)
	if mode == "bow": com = Vectorizing.BagOfWord(com)

	test = np.array(com)
	np.save("test.npy", test)

	test  = np.load("test.npy", allow_pickle=True)
	return test

class DataSet(Dataset):
	def __init__(self, data, train_or_test):
		self.data          = data
		self.train_or_test = train_or_test

	def __len__(self):
		return len(self.data)

	def __getitem__(self, ind):
		if self.train_or_test == 0:
			com = torch.from_numpy(self.data[ind][0]).type(torch.LongTensor)
			lab = torch.tensor    (self.data[ind][1]).type(torch.LongTensor)
			return com, lab

		if self.train_or_test == 1:
			com = torch.from_numpy(self.data[ind]).type(torch.LongTensor)
			return com
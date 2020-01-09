import numpy as np
import operator
from tqdm import tqdm

def Vectorize(data):
	vocb = {}
	with open("Vocabulary.pickle", "rb") as file:
		import pickle
		vocb = pickle.load(file)
		file.close()

	for k, sent in tqdm(enumerate(data), "Vectorization I\t\t"):
		data[k] = np.array([vocb[word] for word in sent if word in vocb])

	max_len = 81
	for k, sent in tqdm(enumerate(data), "Vectorization II\t"):
		data[k] = np.array(sent.tolist() + [0] * (max_len - len(sent)))
	return data

def BagOfWord(data):
	vocb = {}
	with open("Vocabulary.pickle", "rb") as file:
		import pickle
		vocb = pickle.load(file)
		file.close()

	for k, sent in tqdm(enumerate(data), "BagOfWord\t\t"):
		data[k] = np.zeros(len(vocb))
		for word, time in vocb.items():
			if word     in sent: data[k][vocb[word]] = 1
			if word not in sent: data[k][vocb[word]] = 0
	return data
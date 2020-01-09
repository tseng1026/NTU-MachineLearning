import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models import word2vec
from tqdm import tqdm

def Tokenize(data):
	tokn = spacy.load("en_core_web_sm")
	for k, sent in tqdm(enumerate(data), "Tokenization\t\t"):
		data[k] = tokn(sent.lower())
	return data

def Lematize(data):
	lema = spacy.load("en_core_web_sm")
	for k, sent in tqdm(enumerate(data), "Lemmatization\t\t"):
		data[k] = [word.lemma_ for word in sent]
	return data

def Garbages(data):
	garb = ['url', 'user', ',', '.', '!', '#', '?', '-', '“', '”', '/', '@', ':', ';', '-PRON-', '...']
	for k, sent in tqdm(enumerate(data), "Garbages Removing\t"):
		data[k] = [word for word in sent if word not in garb]
	return data

def Stopword(data):
	for k, sent in tqdm(enumerate(data),"Stopwords Removing\t"):
		data[k] = [word for word in sent if word not in STOP_WORDS]
	return data

def Vocabulary(trn, tst, mode="rnn"):
	import os
	import pickle
	import numpy as np
	import pandas as pd

	matx, vocb = [], {}
	if os.path.exists("Vocabulary.pickle"):
		with open("Vocabulary.pickle", "rb") as file:
			vocb = pickle.load(file)
			file.close()
	else:
		trn = pd.read_csv(trn)
		tst = pd.read_csv(tst)
		trn = trn[["comment"]].comment.tolist()
		tst = tst[["comment"]].comment.tolist()
		com = trn + tst

		com = Tokenize(com)
		# com = Lematize(com)
		# com = Garbages(com)
		# com = Stopword(com)

		data = np.array(com)
		modl = word2vec.Word2Vec(data, size=100, min_count=1, workers=5)
		vect = modl.wv

		for k, sent in tqdm(enumerate(data), "Vocabulary\t\t"):
			for word in sent:
				if word not in vocb: matx.append(vect[word])
				if word not in vocb: vocb[word] = len(vocb)
		with open("Vocabulary.pickle", "wb") as file:
			pickle.dump(vocb, file)
			file.close()
		np.save("Embedding.npy", np.array(matx))
	return
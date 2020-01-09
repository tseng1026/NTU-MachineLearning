import numpy as np
import pandas as pd
import torch
from   torch.utils.data import DataLoader
from   sklearn.metrics import f1_score

import Parsing
import Loading
import Preprocessing
import Vectorizing
import Model

if __name__ == "__main__":
	gpu = torch.cuda.is_available()

	# parsing the arguments
	args = Parsing.Args()
	testname = args.t
	modlname = args.m
	outputfile = args.o
	rnn = args.rnn

	test = Loading.LoadTst(testname)
	numb = len(test)

	test = Loading.DataSet(test, 1)
	test = DataLoader(test, batch_size=64, shuffle=False)
	print ("[Done] Segmenting and vectorizing all data!")

	# load done-training model
	matx  = 0
	model = 0
	if rnn == "rnn": matx  = np.load("Embedding.npy")
	if rnn == "bow": model = Model.DNN(18175)
	if rnn == "rnn": model = Model.RNN(matx, "lstm")
	check = torch.load(modlname)
	
	model.load_state_dict(check)
	if gpu: model.cuda()
	print ("[Done] Initializing all model!")

	# set to evaluation mode
	model.eval()

	predt = torch.LongTensor().cuda()
	for ind, com in enumerate(test):
		com = com.cuda()
		out = model(com)
		
		# compute the accuracy value
		pred = torch.max(out, dim=1)[1]
		predt = torch.cat((predt, pred))

	# write the results to file
	index = np.arange(numb)
	index = index.astype("int")

	predict = predt.type(torch.FloatTensor).cpu().numpy().squeeze()
	predict = predict.astype("int")

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(outputfile, header = ["id", "label"], index = None)

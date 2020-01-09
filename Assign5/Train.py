import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader

import Parsing
import Loading
import Preprocessing
import Vectorizing
import Model

if __name__ == "__main__":
	gpu = torch.cuda.is_available()

	# parsing the arguments
	args = Parsing.Args()
	dataname = args.d
	lablname = args.l
	testname = args.t
	modlname = args.m
	rnn = args.rnn

	Preprocessing.Vocabulary(dataname, testname)
	print ("[Done] Constructing vocabulary dictionary and embedding matrix!")

	train, valid = Loading.LoadTrn(dataname, lablname, rnn)
	train = Loading.DataSet(train, 0)
	valid = Loading.DataSet(valid, 0)
	
	train = DataLoader(train, batch_size=64, shuffle=True)
	valid = DataLoader(valid, batch_size=64, shuffle=False)
	print ("[Done] Segmenting and vectorizing all data!")

	# define loss function and optimizer
	matx  = 0
	model = 0
	if rnn == "rnn": matx  = np.load("Embedding.npy")
	if rnn == "bow": model = Model.DNN(18175)
	if rnn == "rnn": model = Model.RNN(matx, "lstm")
	if gpu: model = model.cuda()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
	print ("[Done] Initializing model and all parameters!")

	best = -1
	for epoch in range(100):
		print ("\n###### Epoch: {:d}".format(epoch + 1))

		# set to training mode
		model.train()

		train_loss = []
		train_scre = []
		for ind, (com, lab) in enumerate(train):
			optimizer.zero_grad()

			# preprocess the image data
			if gpu: 
				com = com.cuda()
				lab = lab.cuda()
			out = model(com)

			# compute the loss value
			loss = criterion(out, lab)
			loss.backward()
			train_loss.append(loss.item())

			# compute the f1score
			pred = torch.max(out, dim=1)[1]
			scre = np.mean((lab == pred).cpu().numpy())
			train_scre.append(scre)

			nn.utils.clip_grad_norm_(model.parameters(), 1)
			optimizer.step()

		print ("[Done] Computing train loss: {:.4f}".format(np.mean(train_loss)))
		print ("[Done] Computing train scre: {:.4f}".format(np.mean(train_scre)))

		# set to training mode
		model.eval()

		valid_loss = []
		valid_scre = []
		for ind, (com, lab) in enumerate(valid):

			# preprocess the image data
			if gpu: 
				com = com.cuda()
				lab = lab.cuda()
			out = model(com)

			# compute the loss value
			loss = criterion(out, lab)
			valid_loss.append(loss.item())

			# compute the f1score
			pred = torch.max(out, dim=1)[1]
			scre = np.mean((lab == pred).cpu().numpy())
			valid_scre.append(scre)

		print("[Done] Computing valid loss: {:.4f}".format(np.mean(valid_loss)))
		print("[Done] Computing valid scre: {:.4f}".format(np.mean(valid_scre)))

		# update the best model
		temp = np.mean(valid_scre)	# or considering the valid scre too
		if best <= temp:
			best = temp
			torch.save(model.state_dict(), modlname)

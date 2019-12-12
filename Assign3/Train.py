import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

import Parsing
import Loading
import Model

if __name__=='__main__':
	gpu = torch.cuda.is_available()
	
	# parsing the arguments
	args = Parsing.Args()
	dataname = args.d
	listname = args.l
	modlname = args.m
	mode = args.mode

	data = Loading.LoadTrn(dataname, listname)
	train = data[:20000]
	valid = data[20000:]

	transform = transforms.Compose([
		transforms.ToTensor(),
	])

	train = Loading.DataTrn(train, mode, transform)
	valid = Loading.DataTrn(valid, mode, transform)
	train = DataLoader(train, batch_size=32, shuffle=True)
	valid = DataLoader(valid, batch_size=32, shuffle=False)
	print ("[Done] Loading all data (training and validation)!")

	# define loss function and optimizer
	if mode == "basicmode": model = Model.Baseline()
	if mode == "transpose": model = Model.Transpose()
	if gpu: model = model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
	print ("[Done] Initializing model and all parameters!")

	pltx = []
	trn1, trn2 = [], []
	val1, val2 = [], []
	best = 0.9
	for epoch in range(666):
		print ("\n###### Epoch: {:d}".format(epoch + 1))

		# set to training mode
		model.train()

		train_loss = []
		train_scre = []
		for ind, (img, lab) in enumerate(train):
			optimizer.zero_grad()
			
			# preprocess the image data
			if gpu: 
				img = img.cuda()
				lab = lab.cuda()
			out = model(img)
			
			# compute the loss value
			loss = criterion(out, lab)
			loss.backward()
			train_loss.append(loss.item())

			# compute the accuracy value
			pred = torch.max(out, dim=1)[1]
			scre = np.mean((lab == pred).cpu().numpy())
			train_scre.append(scre)

			optimizer.step()

		print ("[Done] Computing train loss: {:.4f}".format(np.mean(train_loss)))
		print ("[Done] Computing train scre: {:.4f}".format(np.mean(train_scre)))


		# set to training mode
		model.eval()

		valid_loss = []
		valid_scre = []
		for ind, (img, lab) in enumerate(valid):
			
			# preprocess the image data
			if gpu: 
				img = img.cuda()
				lab = lab.cuda()
			out = model(img)
			
			# compute the loss value
			loss = criterion(out, lab)
			valid_loss.append(loss.item())

			# compute the accuracy value
			pred = torch.max(out, dim=1)[1]
			scre = np.mean((lab == pred).cpu().numpy())
			valid_scre.append(scre)


		print("[Done] Computing valid loss: {:.4f}".format(np.mean(valid_loss)))
		print("[Done] Computing valid scre: {:.4f}".format(np.mean(valid_scre)))

		# update the best model
		train_scre = np.mean(train_scre)
		if best < train_scre:
			best = train_scre
			torch.save(model.state_dict(), modlname)


		# plot the graph
		pltx.append(epoch)
		trn1.append(np.mean(train_loss))
		trn2.append(np.mean(train_scre))
		val1.append(np.mean(valid_loss))
		val2.append(np.mean(valid_scre))

	plt.figure()
	plt.xticks(np.arange(0, 110, 10))
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.plot(pltx, trn1, color="red")
	plt.plot(pltx, val1, color="blue")
	plt.savefig("plt1.png")

	plt.figure()
	plt.xticks(np.arange(0, 110, 10))
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.plot(pltx, trn2, color="red")
	plt.plot(pltx, val2, color="blue")
	plt.savefig("plt2.png")
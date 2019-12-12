import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

import Parsing
import Loading
import Model

if __name__=='__main__':
	gpu = torch.cuda.is_available()
	
	# parsing the arguments
	args = Parsing.Args()
	mode = args.mode
	dataname = args.d
	modlname = args.m
	outputfile = args.o

	test = Loading.LoadTst(dataname)

	transform = transforms.Compose([
		transforms.ToTensor(),
		# transforms.Normalize([mean], [std], inplace=False)
	])

	test = Loading.DataTst(test, mode, transform)
	test = DataLoader(test, batch_size=32, shuffle=False)
	print ("[Done] Loading all data (testing)!")

	# load done-training model
	if mode == "basicmode": model = Model.Baseline()
	if mode == "transpose": model = Model.Transpose()
	checkpoint = torch.load(modlname)
	
	model.load_state_dict(checkpoint)
	if gpu: model.cuda()
	print ("[Done] Initializing all model!")


	# set to evaluation mode
	model.eval()

	predt = torch.LongTensor()
	if gpu: predt = predt.cuda()
	for ind, img in enumerate(test):
		if gpu: 
			img = img.cuda()
		out = model(img)
		
		# compute the accuracy value
		pred = torch.max(out, 1)[1]
		predt = torch.cat((predt, pred))

	# write the results to file
	index = np.arange(7000)
	index = index.astype("int")

	predict = predt.type(torch.FloatTensor).cpu().numpy().squeeze()
	predict = predict.astype("int")

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(outputfile, header = ["id", "label"], index = None)

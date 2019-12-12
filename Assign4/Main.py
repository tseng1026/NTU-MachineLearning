import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering, FeatureAgglomeration
from sklearn.mixture import GaussianMixture

import Parsing
import Loading
import Model
import matplotlib.pyplot as plt

if __name__=='__main__':
	gpu = torch.cuda.is_available()
	
	# parsing the arguments
	args = Parsing.Args()
	outputfile = args.o

	dim = args.dim
	clt = args.clt

	train = Loading.LoadData("trainX.npy")
	test  = Loading.LoadData("trainX.npy")

	train = Loading.DataSet(train)
	test  = Loading.DataSet(test)
	train = DataLoader(train, batch_size=32, shuffle=True)
	test  = DataLoader(test,  batch_size=32, shuffle=False)
	print ("[Done] Loading all data (training and testing)!")

	# define loss function and optimizer
	autoencoder = Model.Autoencoder().cuda()

	criterion = nn.MSELoss()
	optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, betas=(0.9, 0.999))
	print ("[Done] Initializing model and all parameters!")

	for epoch in range(15):
		print ("\n###### Epoch: {:d}".format(epoch + 1))

		train_loss = []
		for ind, img in enumerate(train):
			optimizer.zero_grad()
			
			# preprocess the image data
			img = img.cuda()
			encoded, decoded = autoencoder(img)
			
			# compute the loss value
			loss = criterion(decoded, img)
			loss.backward()
			train_loss.append(loss.item())

			optimizer.step()
		print ("[Done] Computing train loss: {:.4f}".format(np.mean(train_loss)))

	with torch.no_grad():
		latents = []
		for ind, img in enumerate(test):
			
			# preprocess the image data
			img = img.cuda()
			encoded, decoded = autoencoder(img)
			
			latents.append(encoded.cpu().detach().numpy())
		latents = np.concatenate(latents, axis = 0)
		latents = latents.reshape((9000, -1))
		latents = (latents - np.mean(latents)) /  np.std(latents)
		latents = (latents - np.min (latents)) / (np.max(latents) - np.min(latents))
	print ("\n[Done] Adopting autoencoder to reduce dimension!")

	# reduce the dimension
	if dim == "agg": latents = FeatureAgglomeration(n_clusters=32).fit(latents).transform(latents)
	if dim == "pca": latents = PCA(n_components=2, whiten=True).fit_transform(latents)
	if dim == "ica": latents = FastICA(n_components=32, random_state=99).fit_transform(latents)
	if dim == "nmf": latents = NMF(n_components=32, random_state=99).fit_transform(latents)
	if dim == "tsne":latents = TSNE(n_components=3, random_state=99, init="pca").fit_transform(latents)
	print ("[Done] Adopting {:s} algorithm to reduce dimension!".format(dim))

	cluster = []
	# compute the cluster
	if clt == "kmeans":   cluster = KMeans(n_clusters=2, random_state=7122).fit(latents)	
	if clt == "spectral": cluster = SpectralClustering(n_clusters=2).fit(latents)
	if clt == "gaussian": cluster = GaussianMixture(n_components=2).fit(latents)
	print ("[Done] Adopting {:s} algorithm to find two cluster!".format(clt))

	predict = []
	if clt == "kmeans":   predict = cluster.fit_predict(latents)
	if clt == "spectral": predict = cluster.fit_predict(latents)
	if clt == "gaussian": predict = cluster.predict(latents)
	print ("[Done] Predict the results!")

	# plot the figure
	label = np.load("trainY.npy")
	tmp = [[], []]
	for k in range(2):
		for i in range(9000):
			if label[i] != k: continue
			tmp[k].append(latents[i])
		tmp[k] = np.array(tmp[k])

	plt.scatter(tmp[0][:,0], tmp[0][:,1], c = "red")
	plt.scatter(tmp[1][:,0], tmp[1][:,1], c = "blue")
	plt.savefig("scatter.png")

	# write the results to file
	index = np.arange(9000)
	index = index.astype("int")
	
	predict = np.array(predict)
	predict = predict.astype("int")

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(outputfile, header = ["id", "label"], index = None)

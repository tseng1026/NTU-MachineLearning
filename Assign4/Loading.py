import os
import numpy as np
import torch
from   torch.utils.data import Dataset
from   torchvision import transforms, datasets

def LoadData(dataname):
	train = np.load(dataname)
	train = train / 255 * 2 - 1
	return train

class DataSet(Dataset):
	def __init__(self, data):
		self.data      = data
		self.transform = transforms.Compose([
						 transforms.ToTensor(),
						 ])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, ind):
		img = self.data[ind]
		img = self.transform(img).type(torch.FloatTensor)
		return img

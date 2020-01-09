import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class DNN(nn.Module):
	def __init__(self, total):
		super(DNN, self).__init__()
		self.first  = nn.Linear(total, 100)
		self.line1 = nn.Sequential(
					 nn.Linear(100, 100),
					 nn.ReLU(),
					 )
		self.line2 = nn.Sequential(
					 nn.Linear(100, 100),
					 nn.ReLU(),
					 )
		self.line3 = nn.Sequential(
					 nn.Linear(100, 100),
					 nn.ReLU(),
					 )
		self.last  = nn.Linear(100, 2)

	def forward(self, com):
		tmp = com.type(torch.cuda.FloatTensor)
		tmp = self.first(tmp)
		tmp = self.line1(tmp)
		tmp = self.line2(tmp)
		tmp = self.line3(tmp)
		mod = self.last (tmp)
		return mod

class RNN(nn.Module):
	def __init__(self, matx, mode):
		super(RNN, self).__init__()
		self.mode = mode
		matx = torch.from_numpy(matx).type(torch.FloatTensor)

		self.embedding = nn.Embedding(matx.size(0), matx.size(1))
		self.embedding.weight = nn.Parameter(matx)

		self.gru  = nn.GRU (input_size=100, hidden_size=100, num_layers=2, dropout=0.3,batch_first=True)
		self.lstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=2, dropout=0.3,batch_first=True)

		self.line1 = nn.Sequential(
					 nn.Linear(100 * 81, 2048),
					 nn.ReLU(),
					 nn.Dropout(0.5),
					 )

		self.line2 = nn.Sequential(
					 nn.Linear(2048, 256),
					 nn.ReLU(),
					 nn.Dropout(0.5),
					 )

		self.line3 = nn.Sequential(
					 nn.Linear(256, 32),
					 nn.ReLU(),
					 nn.Dropout(0.5),
					 )

		self.last = nn.Sequential(
					 nn.Linear(32, 2),
					 nn.Softmax(dim=1),
					 )

	def forward(self, com):
		tmp = self.embedding(com)
		if self.mode == "lstm": tmp, _ = self.lstm(tmp)
		if self.mode == "gru" : tmp, _ = self.gru (tmp)

		tmp = tmp.reshape((com.shape[0], -1))
		tmp = self.line1(tmp)
		tmp = self.line2(tmp)
		tmp = self.line3(tmp)
		mod = self.last (tmp)
		return mod

import torch.nn as nn
import torchvision
import torchvision.models as models

class Baseline(nn.Module):
	def __init__(self):
		super(Baseline, self).__init__()
		self.resnet18 = models.resnet18(pretrained=True)
		self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])

		self.conv1 = nn.Sequential(
					 nn.Conv2d(1, 32, kernel_size=5, padding=2),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(32),
					 nn.MaxPool2d(2),
					 nn.Dropout(0.2),
					 )

		self.conv2 = nn.Sequential(
					 nn.Conv2d(32, 64, kernel_size=3, padding=1),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(64),
					 nn.MaxPool2d(2),
					 nn.Dropout(0.25),
					 )

		self.conv3 = nn.Sequential(
					 nn.Conv2d(64, 128, kernel_size=3,padding=1),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(128),
					 nn.MaxPool2d(2),
					 nn.Dropout(0.3),
					 )

		self.conv4 = nn.Sequential(
					 nn.Conv2d(128, 128, kernel_size=3,padding=1),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(128),
					 nn.MaxPool2d(2),
					 nn.Dropout(0.4),
					 )

		self.conv5 = nn.Sequential(
					 nn.Linear(128*3*3, 256),
					 nn.ReLU(),
					 nn.BatchNorm1d(256),
					 nn.Linear(256, 7),
					 )

	def forward(self, img):
		# tmp = self.resnet18(img)
		tmp = self.conv1(img)
		tmp = self.conv2(tmp)
		tmp = self.conv3(tmp)
		tmp = self.conv4(tmp)

		tmp = tmp.view(-1, 128*3*3)
		mod = self.conv5(tmp)
		return mod


class Transpose(nn.Module):
	def __init__(self):
		super(Transpose, self).__init__()
		self.resnet18 = models.resnet18(pretrained=True)
		self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])

		self.conv1 = nn.Sequential(
					 nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(256),
					 nn.MaxPool2d(2),
					 nn.Dropout(0.2),
					 )

		self.conv2 = nn.Sequential(
					 nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(128),
					 nn.MaxPool2d(2),
					 nn.Dropout(0.25),
					 )

		self.conv3 = nn.Sequential(
					 nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(64),
					 nn.MaxPool2d(2),
					 nn.Dropout(0.3),
					 )

		self.conv4 = nn.Sequential(
					 nn.ConvTranspose2d( 64,  32, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(32),
					 nn.MaxPool2d(2),
					 nn.Dropout(0.4),
					 )

		self.conv5 = nn.Sequential(
					 nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=True),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(64),
					 nn.Dropout(0.5),
					 )

		self.conv6 = nn.Sequential(
					 nn.Linear(64*2*2, 256),
					 nn.ReLU(),
					 nn.BatchNorm1d(256),
					 nn.Linear(256, 7),
					 )

	def forward(self, img):
		tmp = self.resnet18(img)
		tmp = self.conv1(tmp)
		tmp = self.conv2(tmp)
		tmp = self.conv3(tmp)
		tmp = self.conv4(tmp)
		tmp = self.conv5(tmp)

		tmp = tmp.view(-1, 64*2*2)
		mod = self.conv6(tmp)
		return mod

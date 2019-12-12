import torch.nn as nn
import torchvision
import torchvision.models as models

KERNEL=4
STRIDE=2

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()

		self.relu  = nn.LeakyReLU()

		self.conv1 = nn.Conv2d(   3, 1024, kernel_size=3, stride=STRIDE, padding=1, bias=True)
		self.conv2 = nn.Conv2d(1024,  256, kernel_size=3, stride=STRIDE, padding=1, bias=True)
		self.conv3 = nn.Conv2d( 256,   64, kernel_size=3, stride=STRIDE, padding=1, bias=True)
		self.conv4 = nn.Conv2d(  64,   16, kernel_size=3, stride=STRIDE, padding=1, bias=True)
		self.conv5 = nn.Conv2d(  16,    8, kernel_size=3, stride=STRIDE, padding=1, bias=True)

		self.tran5 = nn.ConvTranspose2d(   8,   16, kernel_size=KERNEL, stride=STRIDE, padding=1, bias=True)
		self.tran4 = nn.ConvTranspose2d(  16,   64, kernel_size=KERNEL, stride=STRIDE, padding=1, bias=True)
		self.tran3 = nn.ConvTranspose2d(  64,  256, kernel_size=KERNEL, stride=STRIDE, padding=1, bias=True)
		self.tran2 = nn.ConvTranspose2d( 256, 1024, kernel_size=KERNEL, stride=STRIDE, padding=1, bias=True)
		self.tran1 = nn.ConvTranspose2d(1024,    3, kernel_size=KERNEL, stride=STRIDE, padding=1, bias=True)

	def forward(self, img):
		# encoder
		tmp = self.relu(self.conv1(img))		# (N, 1024, 16, 16)
		tmp = self.relu(self.conv2(tmp))		# (N,  256,  8,  8)
		tmp = self.relu(self.conv3(tmp))		# (N,   64,  4,  4)
		tmp = self.relu(self.conv4(tmp))		# (N,   16,  2,  2)
		tmp = self.relu(self.conv5(tmp))		# (N,    8,  1,  1)
		encoded = tmp

		# decoder
		tmp = self.tran5(self.relu(tmp))		# (N,   16,  2,  2)
		tmp = self.tran4(self.relu(tmp))		# (N,   64,  4,  4)
		tmp = self.tran3(self.relu(tmp))		# (N,  256,  8,  8)
		tmp = self.tran2(self.relu(tmp))		# (N, 1024, 16, 16)
		tmp = self.tran1(self.relu(tmp))		# (N,    3, 32, 32)
		decoded = tmp

		return encoded, decoded

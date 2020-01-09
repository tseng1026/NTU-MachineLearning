import os
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
	warntext = "warning.txt"
	f = open(warntext, "r")

	k = 0
	pltx = []
	trn1 = []
	trn2 = []
	val1 = []
	val2 = []
	cont = f.readlines()
	for k in range(100):
		pltx.append(int(cont[k * 6 + 1][14:]))
		trn1.append(float(cont[k * 6 + 2][29:]))
		trn2.append(float(cont[k * 6 + 3][29:]))
		val1.append(float(cont[k * 6 + 4][29:]))
		val2.append(float(cont[k * 6 + 5][29:]))


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
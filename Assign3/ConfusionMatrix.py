import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

if __name__=='__main__':
	labl = pd.read_csv("train.csv")
	pred = pd.read_csv("prediction.csv")

	labl = labl.iloc[:,1].values.tolist()
	pred = pred.iloc[:,1].values.tolist()

	labl = np.array(labl)
	pred = np.array(pred)

	matrix = confusion_matrix(labl, pred).astype("float32")
	for k in range(7):
		matrix[k] = matrix[k] / np.sum(matrix[k])
	matrix = np.around(matrix, decimals=4)
	matrix = pd.DataFrame(matrix, index=range(0, 7), columns=range(0, 7))
	
	plt.figure()
	sn.set(font_scale=0.8)
	plt.title("Confusion Matrix")
	plt.xlabel("Prediction")
	plt.ylabel("Ground Truth")
	ax = sn.heatmap(matrix, annot=True,annot_kws={"size": 8}, cmap="binary", fmt=".4f", square=True)
	ax.set_ylim(7.0, 0)
	plt.savefig("ConfusionMatrix.png")


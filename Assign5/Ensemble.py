import numpy as np
import pandas as pd

import Parsing

if __name__ == "__main__":
	# parsing the arguments
	args = Parsing.Args()
	outputfile = args.o

	# sum of the results	
	total = np.zeros((860, ))
	for k in range(7):
		pred = pd.read_csv("prediction" + str(k+1) + ".csv")
		pred = pred[["label"]].label.tolist()
		pred = np.array(pred)

		total += pred

	# write the results to file
	index = np.arange(860)
	index = index.astype("int")

	predict = total // 4
	predict = predict.astype("int")

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(outputfile, header = ["id", "label"], index = None)
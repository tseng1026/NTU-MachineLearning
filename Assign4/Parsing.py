import argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", default="./prediction.csv")
	parser.add_argument("--dim", default="ica")
	parser.add_argument("--clt", default="kmeans")

	args = parser.parse_args()
	return args

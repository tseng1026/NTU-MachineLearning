import argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", default="./train_x.csv")
	parser.add_argument("-l", default="./train_y.csv")
	parser.add_argument("-t", default="./test_x.csv")
	parser.add_argument("-m", default="./model_best.pth.tar")
	parser.add_argument("-o", default="./prediction.csv")
	parser.add_argument("--rnn", default="rnn")

	args = parser.parse_args()
	return args

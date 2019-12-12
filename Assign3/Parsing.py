import argparse

def Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", default="./train_img/")
	parser.add_argument("-l", default="./train.csv")
	parser.add_argument("-w", default="./warning.txt")
	parser.add_argument("-m", default="./model_best.pth")
	parser.add_argument("-o", default="./prediction.csv")
	parser.add_argument("--mode", default="transpose")

	args = parser.parse_args()
	return args

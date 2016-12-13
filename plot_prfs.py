from utilities import plot_prfs,
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="plot_prfs.py: used to \
		plot our precision, recall and F1 data")
	parser.add_argument(
		"-r",
		"--prfs-train",
		metavar="FILEPATH",
		type=str,
		help="filepath to CSV of precision, recall and F1 performance of \
		training set"
		)
	parser.add_argument(
		"-e",
		"--prfs-test",
		metavar="FILEPATH",
		type=str,
		help="filepath to CSV of precision, recall and F1 performance of \
		test set"
		)

	args = parser.parse_args()
	if (args.prfs_train is not None and args.prfs_test is  None):
		raise ValueError("Filepath for train data provided, but no filepath \
			for test data")
	elif (args.prfs_test is not None and args.prfs_train is  None):
		raise ValueError("Filepath for test data provided, but no filepath \
			for train data")
	return args

def read_prfs_file(prfs_file):
	prfs_data = []
	labels = None
	with open(prfs_file,"r") as f:
		i = 0
		for line in f:
			line = line.strip()
			if line == "":
				continue
			row = line.split(",")
			if i == 0:
				labels = row
			else:
				prfs_data.append(row)
			i += 1
	return prfs_data,labels


def main():
	args = parse_args()
	if args.prfs_train is not None:
		prfs_train,labels_train = read_prfs_file(args.prfs_train)
		prfs_test,labels_test = read_prfs_file(args.prfs_test)
		for i in range(len(labels_train)):
			if labels_train[i] != labels_test[i]:
				raise ValueError("Label headers do not match!")
		plot_prfs(labels_train,prfs_train,prfs_test)

main()
import plotly
import plotly.graph_objs as go
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

def sort_prfs_data(labels,sorted_labels,prfs_data):
	positions = []
	for label in sorted_labels:
		positions.append(labels.index(label))
	new_data = []
	for i in range(4):
		new_data.append([])
		for j in range(20):
			new_data[i].append(0)
	i = 0
	for pos in positions:
		for j in range(4):
			new_data[j][i] = prfs_data[j][pos]
		i += 1
	return new_data

def plot_prfs(t_labels,prfs_train,prfs_test):
	sorted_labels = sorted(t_labels)
	prfs_train = sort_prfs_data(t_labels,sorted_labels,prfs_train)
	prfs_test = sort_prfs_data(t_labels,sorted_labels,prfs_test)
	t_labels = sorted_labels
	
	itf_tuples = [
	(0,"Precision - Training data",'precision_train.html'),
	(1,"Recall - Training data",'recall_train.html'),
	(2,"F1-score - Training data",'f1_train.html'),
	(3,"Number of Samples - Training data",'samples_train.html'),
	(0,"Precision - Test data",'precision_test.html'),
	(1,"Recall - Test data",'recall_test.html'),
	(2,"F1-score - Test data",'f1_test.html'),
	(3,"Number of Samples - Test data",'samples_test.html'),
	]

	j = 0
	for index,title,fname in itf_tuples:
		y_plt = []
		for i in range(len(t_labels)):
			if j < 4:
				y_plt.append(prfs_train[index][i])
			else:
				y_plt.append(prfs_test[index][i])
		bar = go.Bar(x=t_labels,y=y_plt)
		data=[bar]
		layout = go.Layout(title=title)
		fig = go.Figure(data=data, layout=layout)
		# import IPython; IPython.embed()
		plotly.offline.plot(fig,filename=fname)
		j += 1

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
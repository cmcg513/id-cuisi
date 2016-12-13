import plotly
import plotly.graph_objs as go

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
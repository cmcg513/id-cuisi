# Author: Casey McGinley <cmm771@nyu.edu>
# Machine Learning, Fall 2016
# Final Project: Identifying Cultural Cuisine Influences in Recipes
# 
# baseline.py
# This script represents my first attempt at developing an effective tool to 
# identify/classify a recipe's type(s) of cuisines. This script will be the
# baseline from which I attempt to improve future tools.

import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import metrics
import argparse
import plotly
import plotly.graph_objs as go

# sets up basic cmd line parsing
def parse_args():
	parser = argparse.ArgumentParser(description="baseline.py: uses a \
		straightforward Naive Bayes classification method on recipe data \
		and reports some statistics/metrics on its performance")
	parser.add_argument(
		"-f",
		"--file",
		metavar="FILEPATH",
		help="the filepath to the dataset; defaults to relative path: \
		data/train.json"
		)

	return parser.parse_args()

# given a set of recipes, it returns the transformed data, the list of expected 
# labels (cuisines), a CV instance fitted to the given data and ready to 
# transform future data, and the target labels
def transform_and_fit(recipes):
	# get a unique list of all possible cuisines; the labels for out targets
	t_labels = list(set(recipes['cuisine']))

	# generate a list of ints where each int represents an index into t_labels
	# this is the list of targets, the numeric form of the expected cuisine of 
	# each recipe
	t_nums = []
	for c in recipes['cuisine']:
		t_nums.append(t_labels.index(c))

	# create a CV instance
	# given a string, it will create a numeric representation indicating which 
	# words are and are not present (see binary=True)
	c_vect = CountVectorizer(binary=True)

	# collapse each ingredient list to single string of space-separated words
	data = []
	for i_list in recipes['ingredients']:
		data.append(" ".join(i_list))

	# fit the CV instance to our data and transform our data into a form 
	# suitable for analysis with scikit-learns modules
	X = c_vect.fit_transform(data)

	return X,t_nums,t_labels,c_vect

# given a set of recipes, it returns the transformed dataand the list of 
# expected labels (cuisines)
# 
# NOTE: this is basically the same procedure as transform_and_fit() except
# that it utilizes the label ordering and fitted CV instance returned by 
# transform_and_fit() instead of generating this on its own
def transform(recipes,t_labels,c_vect):
	t_nums = []
	for c in recipes['cuisine']:
		t_nums.append(t_labels.index(c))
	
	data = []
	for i_list in recipes['ingredients']:
		data.append(" ".join(i_list))

	X = c_vect.transform(data)

	return X,t_nums

def process_data(filepath):
	# read in the data set
	full_data_set = pd.read_json(filepath)
	
	# split the data into train and test portions
	train,test = model_selection.train_test_split(full_data_set)
	
	# report the size of the data set
	print("Full dataset size: {0}".format(len(full_data_set)))
	print("Train dataset size: {0} ({1:2.2f}%)".format(len(train),float(len(train))/float(len(full_data_set))*100))
	print("Test dataset size: {0} ({1:2.2f}%)".format(len(test),float(len(test))/float(len(full_data_set))*100))

	# fit the CV (c_vect) to the train data and transform it into the appropriate 
	# format (X), acquiring the expected targets (Y) as well
	X,Y,t_labels,c_vect = transform_and_fit(train)

	# initialize an instance of a Bernoulli Naive Bayes model
	bnb = BernoulliNB()

	# fit the Bernoulli model to the train data, giving us a classifier
	classifier = bnb.fit(X,Y)

	# transform the test data
	X_test,Y_test = transform(test,t_labels,c_vect)
	
	# get the list of predicted targets on the train and test data
	train_pred = classifier.predict(X)
	test_pred = classifier.predict(X_test)

	# compute the accuracy of the predictions
	train_acc = np.mean(train_pred == Y)
	test_acc = np.mean(test_pred == Y_test)

	# report the accuracy
	print("Training Set accuracy: {0}".format(train_acc))
	# print("Training Set f1-score: {0}".format(metrics.f1_score(Y,train_pred)))
	print("Testing Set accuracy: {0}".format(test_acc))
	# print("Testing Set f1-score: {0}".format(metrics.f1_score(Y_test,test_pred)))

	# report additional metrics
	print(metrics.classification_report(Y,train_pred,target_names=t_labels))
	print(metrics.classification_report(Y_test,test_pred,target_names=t_labels))
	prfs_train = metrics.precision_recall_fscore_support(Y,train_pred)
	prfs_test = metrics.precision_recall_fscore_support(Y_test,test_pred)
	c_matrix_train = metrics.confusion_matrix(Y,train_pred)
	c_matrix_test = metrics.confusion_matrix(Y_test,test_pred)

	f = open("prfs_train.csv","w")
	for i in prfs_train:
		f.write(",".join([str(x) for x in list(i)])+"\n")
	f.close()
	f = open("prfs_test.csv","w")
	for i in prfs_test:
		f.write(",".join([str(x) for x in list(i)])+"\n")
	f.close()

	f = open("c_matrix_train.csv","w")
	for i in c_matrix_train:
		f.write(",".join([str(x) for x in list(i)])+"\n")
	f.close()

	f = open("c_matrix_test.csv","w")
	for i in c_matrix_test:
		f.write(",".join([str(x) for x in list(i)])+"\n")
	f.close()

	y_plt = []
	for i in range(len(t_labels)):
		y_plt.append(prfs_train[0][i])
	bar = go.Bar(x=t_labels,y=y_plt)
	data=[bar]
	layout = go.Layout(title="Precision - Training data")
	fig = go.Figure(data=data, layout=layout)
	# import IPython; IPython.embed()
	plotly.offline.plot(fig,filename='precision_train.html')

	y_plt = []
	for i in range(len(t_labels)):
		y_plt.append(prfs_train[1][i])
	bar = go.Bar(x=t_labels,y=y_plt)
	data=[bar]
	layout = go.Layout(title="Recall - Training data")
	fig = go.Figure(data=data, layout=layout)
	# import IPython; IPython.embed()
	plotly.offline.plot(fig,filename='recall_train.html')

	y_plt = []
	for i in range(len(t_labels)):
		y_plt.append(prfs_train[2][i])
	bar = go.Bar(x=t_labels,y=y_plt)
	data=[bar]
	layout = go.Layout(title="F1-score - Training data")
	fig = go.Figure(data=data, layout=layout)
	# import IPython; IPython.embed()
	plotly.offline.plot(fig,filename='f1_train.html')

	y_plt = []
	for i in range(len(t_labels)):
		y_plt.append(prfs_train[3][i])
	bar = go.Bar(x=t_labels,y=y_plt)
	data=[bar]
	layout = go.Layout(title="Number of Samples - Training data")
	fig = go.Figure(data=data, layout=layout)
	# import IPython; IPython.embed()
	plotly.offline.plot(fig,filename='samples_train.html')

	y_plt = []
	for i in range(len(t_labels)):
		y_plt.append(prfs_test[0][i])
	bar = go.Bar(x=t_labels,y=y_plt)
	data=[bar]
	layout = go.Layout(title="Precision - Test data")
	fig = go.Figure(data=data, layout=layout)
	# import IPython; IPython.embed()
	plotly.offline.plot(fig,filename='precision_test.html')

	y_plt = []
	for i in range(len(t_labels)):
		y_plt.append(prfs_test[1][i])
	bar = go.Bar(x=t_labels,y=y_plt)
	data=[bar]
	layout = go.Layout(title="Recall - Test data")
	fig = go.Figure(data=data, layout=layout)
	# import IPython; IPython.embed()
	plotly.offline.plot(fig,filename='recall_test.html')

	y_plt = []
	for i in range(len(t_labels)):
		y_plt.append(prfs_test[2][i])
	bar = go.Bar(x=t_labels,y=y_plt)
	data=[bar]
	layout = go.Layout(title="F1-score - Test data")
	fig = go.Figure(data=data, layout=layout)
	# import IPython; IPython.embed()
	plotly.offline.plot(fig,filename='f1_test.html')

	y_plt = []
	for i in range(len(t_labels)):
		y_plt.append(prfs_test[3][i])
	bar = go.Bar(x=t_labels,y=y_plt)
	data=[bar]
	layout = go.Layout(title="Number of Samples - Test data")
	fig = go.Figure(data=data, layout=layout)
	# import IPython; IPython.embed()
	plotly.offline.plot(fig,filename='samples_test.html')


# the main routine; grabs command line args and sets the program in motion
def main():
	args = parse_args()

	# use a default filepath if unspecified
	if args.file:
		process_data(args.file)
	else:
		process_data("data/train.json")

main()
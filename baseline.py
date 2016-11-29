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

def main():
	json = pd.read_json("train.json")
	train,test = model_selection.train_test_split(json)
	print("Train dataset size: {0} ({1:2.2f}%)".format(len(train),float(len(train))/float(len(json))*100))
	print("Test dataset size: {0} ({1:2.2f}%)".format(len(test),float(len(test))/float(len(json))*100))

	X,Y,t_labels,c_vect = transform_and_fit(train)

	bnb = BernoulliNB()
	clf=bnb.fit(X,Y)
	X_test,Y_test=transform(test,t_labels,c_vect)
	train_pred = clf.predict(X)
	test_pred = clf.predict(X_test)
	train_acc = np.mean(train_pred == Y)
	test_acc = np.mean(test_pred == Y_test)
	print("Training Set accuracy: {0}".format(train_acc))
	print("Testing Set accuracy: {0}".format(test_acc))
	print(metrics.classification_report(Y,train_pred,target_names=t_labels))
	print(metrics.confusion_matrix(Y,train_pred))
	# print("\n"); import IPython; IPython.embed()

main()
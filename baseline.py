import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import metrics

def process_first(raw_data):
	targets = list(set(raw_data['cuisine']))
	n_targs = []
	for x in raw_data['cuisine']:
		n_targs.append(targets.index(x))

	ingredients = set()
	for l in raw_data['ingredients']:
		for x in l:
			ingredients.add(x)
	ingredients = list(ingredients)

	count_vect = CountVectorizer(binary=True)

	data = []
	for d in raw_data['ingredients']:
		data.append(" ".join(d))

	X = count_vect.fit_transform(data)

	return X, n_targs,targets,count_vect,ingredients

def process_additional(raw_data,targets,count_vect,ingredients):
	n_targs = []
	for x in raw_data['cuisine']:
		n_targs.append(targets.index(x))
	
	data = []
	for d in raw_data['ingredients']:
		data.append(" ".join(d))

	X = count_vect.transform(data)

	return X,n_targs

def main():
	json = pd.read_json("train.json")
	train,test = model_selection.train_test_split(json)
	print("Train dataset size: {0} ({1:2.2f}%)".format(len(train),float(len(train))/float(len(json))*100))
	print("Test dataset size: {0} ({1:2.2f}%)".format(len(test),float(len(test))/float(len(json))*100))

	X,Y,targets,count_vect,ingredients = process_first(train)

	# X = []

	# print("Total samples: {0}".format(len(train['ingredients'])))
	# c = 0
	# for d in train['ingredients']:
	# 	if c % 1000 == 0:
	# 		print(c)
	# 	x = []
	# 	for i in ingredients:
	# 		if i in d:
	# 			x.append(1)
	# 		else:
	# 			x.append(0)
	# 	X.append(x)
	# 	c += 1

	bnb = BernoulliNB()
	clf=bnb.fit(X,Y)
	X_test,Y_test=process_additional(test,targets,count_vect,ingredients)
	train_pred = clf.predict(X)
	test_pred = clf.predict(X_test)
	train_acc = np.mean(train_pred == Y)
	test_acc = np.mean(test_pred == Y_test)
	print("Training Set accuracy: {0}".format(train_acc))
	print("Testing Set accuracy: {0}".format(test_acc))
	print(metrics.classification_report(Y,train_pred,target_names=targets))
	print(metrics.confusion_matrix(Y,train_pred))
	# print("\n"); import IPython; IPython.embed()

main()
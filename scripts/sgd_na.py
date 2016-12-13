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
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import metrics
import argparse

from utilities import plot_prfs

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, SGDClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

PLOT = False
MODEL_DEFAULT = 11
ATOMIC_INGREDIENTS = False

# sets up basic cmd line parsing
def parse_args():
	parser = argparse.ArgumentParser(description="baseline.py: uses a \
		straightforward Naive Bayes classification method on recipe data \
		and reports some statistics/metrics on its performance")
	parser.add_argument(
		"-f",
		"--file",
		metavar="FILEPATH",
		type=str,
		help="the filepath to the dataset; defaults to relative path: \
		data/train.json"
		)
	parser.add_argument(
		"-m",
		"--model",
		metavar="MODEL-NUM",
		type=int,
		help="the number indicating which classifier model to use; default is \
		Bernoulli Naive Bayes"
		)
	parser.add_argument(
		"-p",
		"--plot",
		action="store_true",
		help="flag indicating plots should be generated"
		)
	parser.add_argument(
		"-r",
		"--random",
		metavar="SEED",
		type=int,
		help="specify the random seed (optional)"
		)
	parser.add_argument(
		"-t",
		"--tokenize-whole-ingredients",
		action="store_true",
		help="flag indicating that each ingredient should be atomic (e.g. \"garlic salt\" should not be split into two separate tokens)"
		)

	return parser.parse_args()

def get_Xandy(recipes,t_labels):
	t_nums = []
	for c in recipes['cuisine']:
		t_nums.append(t_labels.index(c))

	data = []
	for i_list in recipes['ingredients']:
		new_list = []
		for i in i_list:
			if ATOMIC_INGREDIENTS:
				new_list.append(i.replace(" ","-"))
			else:
				new_list.append(i)
		data.append(" ".join(new_list))
	return data,t_nums

# given a set of recipes, it returns the transformed data, the list of expected 
# labels (cuisines), a CV instance fitted to the given data and ready to 
# transform future data, and the target labels
def transform_and_fit(recipes):
	# get a unique list of all possible cuisines; the labels for out targets
	t_labels = list(set(recipes['cuisine']))

	# generate a list of ints where each int represents an index into t_labels
	# this is the list of targets, the numeric form of the expected cuisine of 
	# each recipe

	data,t_nums = get_Xandy(recipes,t_labels)
	# create a CV instance
	# given a string, it will create a numeric representation indicating which 
	# words are and are not present (see binary=True)
	# c_vect = CountVectorizer(binary=True)
	# tf_trans = TfidfTransformer(use_idf=False)

	# collapse each ingredient list to single string of space-separated words
	

	# fit the CV instance to our data and transform our data into a form 
	# suitable for analysis with scikit-learns modules
	# X = c_vect.fit_transform(data)
	# X = tf_trans.fit_transform(X)

	return data,t_nums,t_labels#,c_vect,tf_trans #X,

# given a set of recipes, it returns the transformed dataand the list of 
# expected labels (cuisines)
# 
# NOTE: this is basically the same procedure as transform_and_fit() except
# that it utilizes the label ordering and fitted CV instance returned by 
# transform_and_fit() instead of generating this on its own
def transform(recipes,t_labels):#,c_vect,tf_trans):
	data,t_nums = get_Xandy(recipes,t_labels)

	# X = c_vect.transform(data)
	# X = tf_trans.transform(X)

	return data,t_nums

def get_model(model_num,seed):
	params = {}
	if model_num == 0:
		model_string = "RidgeClassifierCV"
		model = RidgeClassifierCV()
	elif model_num == 1:
		model_string = "LogisticRegression"
		# model = LogisticRegression(C=75)
		model = LogisticRegression()
		params['clf__C'] = [10] # default: 1; small vals stronger regulariuzation
		params['clf__penalty'] = ['l2'] # newton-cg, sag and lbfgs only use l2; default: l2
		params['clf__dual'] = [False] # default: False; only for l2 and only good when n_samples > n_features
		params['clf__fit_intercept'] = [True] # default: True; should constant be added to decision function
		params['clf__class_weight'] = [None] # default: None
		params['clf__max_iter'] = [100] # default: 100; only for newton, sag and lbfgs
		params['clf__solver'] = ['lbfgs']#['liblinear','lbfgs','newton-cg','sag']#['lbfgs'] # default: liblinear; newton, sag an lbfgs might be better for multiclass problems (multinomial loss?)
		params['clf__tol'] = [1e-4]#,1e-3] #default: 1e-4; tolerance for stopping
		params['clf__multi_class'] = ['ovr'] # default: ovr; ovr uses binary problem to fit; multinomial fits on multinomial minimizing loss across entire prob distribution; newton, sag and lbfgs only	
		params['clf__random_state'] =[seed]
	elif model_num == 2:
		model_string = "DecisionTreeClassifier"
		model = DecisionTreeClassifier(max_depth=30,random_state=seed)
	elif model_num == 3: 
		model_string = "RandomForestClassifier"
		model = RandomForestClassifier(random_state=seed)
	elif model_num == 4:
		model_string = "GradientBoostingClassifier"
		model = GradientBoostingClassifier(n_estimators=175, learning_rate=0.085, max_depth=3, random_state=seed)
	elif model_num == 5:
		model_string = "AdaBoostClassifier"
		model = AdaBoostClassifier(random_state=seed)
	elif model_num == 6:
		model_string = "GradientBoostingRegressor"
		model = GradientBoostingRegressor(random_state=seed)
	elif model_num == 7:
		model_string = "ExtraTreesClassifier"
		model = ExtraTreesClassifier(random_state=seed)
	elif model_num == 8:
		model_string = "ExtraTreesRegressor"
		model = ExtraTreesRegressor(random_state=seed)
	elif model_num == 9:
		model_string = "KNeighborsClassifier"
		model = KNeighborsClassifier(n_neighbors=3)
	elif model_num == 10:
		model_string = "SVC"
		model = SVC(random_state=seed)
	elif model_num == 11:
		model_string = "BernoulliNB"
		model = BernoulliNB()
	elif model_num == 12:
		model_string = "MultinomialNB"
		model = MultinomialNB()
	elif model_num == 13:
		model_string = "SGDClassifier"
		model = SGDClassifier()
		params['clf__random_state'] = [seed]
		params['clf__loss'] =['modified_huber']#,'log']#,'squared_hinge','perceptron','hinge']
		params['clf__penalty'] = ['elasticnet']#['none','l2','l1','elasticnet']
		params['clf__alpha'] = [1e-4]#[1e-3,1e-4,1e-5]
		params['clf__class_weight'] = [None]#,'balanced']
	elif model_num == 14:
		model_string = "VotingClassifier"
		logr = LogisticRegression(C=10,random_state=seed)
		# rdge = RidgeClassifierCV()
		rfor = RandomForestClassifier(random_state=seed)
		grad = GradientBoostingClassifier(random_state=seed)
		adac = AdaBoostClassifier(random_state=seed)
		etcl = ExtraTreesClassifier(random_state=seed)
		dect = DecisionTreeClassifier(random_state=seed)
		# grdr = GradientBoostingRegressor()
		# etrg = ExtraTreesRegressor()
		# kngh = KNeighborsClassifier(n_neighbors=5)
		model = VotingClassifier(estimators=[('lr',logr),('rfc',rfor),('gbc',grad),('abc',adac),('etc',etcl),('dtc',dect)],voting='soft',weights=[5.02,5.44,6,5.96,5.36,4.68])

	return model,model_string,params

def simple_tokenizer(x):
	return x.split(" ")

def process_data(filepath,model_num,seed):
	# read in the data set
	full_data_set = pd.read_json(filepath)
	
	# split the data into train and test portions
	train,test = model_selection.train_test_split(full_data_set,random_state=seed)
	
	# report the size of the data set
	print("Full dataset size: {0}".format(len(full_data_set)))
	print("Train dataset size: {0} ({1:2.2f}%)".format(len(train),float(len(train))/float(len(full_data_set))*100))
	print("Test dataset size: {0} ({1:2.2f}%)".format(len(test),float(len(test))/float(len(full_data_set))*100))

	# fit the CV (c_vect) to the train data and transform it into the appropriate 
	# format (X), acquiring the expected targets (y) as well
	print("Transforming training data...")
	X_train,y_train,t_labels = transform_and_fit(train) #,c_vect,tf_trans
	# print(X_train)
	# initialize an instance of a model
	model,model_string,params = get_model(model_num,seed)
	print("Model used: "+model_string)

	# create Pipeline
	pline = Pipeline([
		('vect', CountVectorizer(tokenizer=simple_tokenizer)),
		('tfidf', TfidfTransformer()),
		('clf', model)
		])

	params['vect__tokenizer'] = [None]#[simple_tokenizer,None]
	params['vect__ngram_range'] = [(1,2)]#,(1,2)]
	params['vect__binary'] = [True]#,False]
	params['tfidf__use_idf'] = [True]#[True,False]
	params['tfidf__norm'] = ['l2']#['l1','l2',None]
	params['tfidf__smooth_idf'] = [True]#[True,False]
	params['tfidf__sublinear_tf'] = [True]#[True,False]
	params['vect__max_df'] = [.6]#,.7,.9]
	# params['vect__stop_words'] = ['english',None]

	gs_clf = GridSearchCV(pline,params,n_jobs=3)

	# fit the Bernoulli model to the train data, giving us a classifier
	print("Fitting model...")
	# density_problem = False
	# try:
	# classifier = model.fit(X,y)
	gs_clf.fit(X_train,y_train)

	print("Best params (train)")
	print(gs_clf.best_params_)
	print("Best score: "+str(gs_clf.best_score_))
	print("Score: "+str(gs_clf.score(X_train,y_train)))

	# except:
	# 	X = X.toarray()
	# 	y = y.toarray()
	# 	classifier = model.fit(X,y)
	# 	density_problem = True
	# transform the test data
	print("Transforming test data...")
	X_test,y_test = transform(test,t_labels)#,c_vect,tf_trans)
	# if density_problem:
	# 	X_test = X_test.toarray()
	# 	y_test = y_test.toarray()
	# get the list of predicted targets on the train and test data
	print("Predicting train data...")
	train_pred = gs_clf.predict(X_train)
	print("Predicting test data...")
	test_pred = gs_clf.predict(X_test)

	# compute the accuracy of the predictions
	train_acc = np.mean(train_pred == y_train)
	test_acc = np.mean(test_pred == y_test)

	# report the accuracy
	print("Best accuracy (train): {0}".format(train_acc))
	# print("Training Set f1-score: {0}".format(metrics.f1_score(y,train_pred)))
	print("Accuracy (train): {0}".format(test_acc))
	# print("Testing Set f1-score: {0}".format(metrics.f1_score(y_test,test_pred)))

	# report ROC score
	# print ("Training set ROC score: "+str(np.mean(cross_val_score(classifier, X, y, scoring='roc_auc'))))
	# print ("Testing set ROC score: "+str(np.mean(cross_val_score(classifier, X_test, y_test, scoring='roc_auc'))))
	
	# report additional metrics
	print(metrics.classification_report(y_train,train_pred,target_names=t_labels))
	print(metrics.classification_report(y_test,test_pred,target_names=t_labels))
	prfs_train = metrics.precision_recall_fscore_support(y_train,train_pred)
	prfs_test = metrics.precision_recall_fscore_support(y_test,test_pred)
	c_matrix_train = metrics.confusion_matrix(y_train,train_pred)
	c_matrix_test = metrics.confusion_matrix(y_test,test_pred)

	f = open("prfs_train.csv","w")
	f.write(",".join(t_labels)+"\n")
	for i in prfs_train:
		f.write(",".join([str(x) for x in list(i)])+"\n")
	f.close()
	f = open("prfs_test.csv","w")
	f.write(",".join(t_labels)+"\n")
	for i in prfs_test:
		f.write(",".join([str(x) for x in list(i)])+"\n")
	f.close()

	f = open("c_matrix_train.csv","w")
	f.write(",".join(t_labels)+"\n")
	for i in c_matrix_train:
		f.write(",".join([str(x) for x in list(i)])+"\n")
	f.close()

	f = open("c_matrix_test.csv","w")
	f.write(",".join(t_labels)+"\n")
	for i in c_matrix_test:
		f.write(",".join([str(x) for x in list(i)])+"\n")
	f.close()

	if PLOT:
		plot_prfs(t_labels,prfs_train,prfs_test)


# the main routine; grabs command line args and sets the program in motion
def main():
	global PLOT, ATOMIC_INGREDIENTS
	args = parse_args()

	PLOT = args.plot
	if args.model is None:
		args.model = MODEL_DEFAULT
	if args.tokenize_whole_ingredients is not None:
		ATOMIC_INGREDIENTS = args.tokenize_whole_ingredients

	# use a default filepath if unspecified
	if args.file:
		process_data(args.file,args.model,args.random)
	else:
		process_data("data/train.json",args.model,args.random)

main()
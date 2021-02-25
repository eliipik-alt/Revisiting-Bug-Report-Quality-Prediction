import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict



def test (X, y):


	tfidfconverter = TfidfTransformer()
	X = tfidfconverter.fit_transform(X).toarray()
	
	kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']
	# A function which returns the corresponding SVC model
	def getClassifier(ktype):
	    if ktype == 0:
	        # Polynomial kernal
	        return SVC(kernel='poly', degree=8, gamma="auto", probability=True)
	    elif ktype == 1:
	        # Radial Basis Function kernal
	        return SVC(kernel='rbf', gamma="auto", probability=True)
	    elif ktype == 2:
	        # Sigmoid kernal
	        return SVC(kernel='sigmoid', gamma="auto", probability=True)
	    elif ktype == 3:
	        # Linear kernal
	        return SVC(kernel='linear', gamma="auto", probability=True)

	# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
	param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['sigmoid']}

	svclassifier = getClassifier(2)


	grid = GridSearchCV(SVC(probability=True),param_grid,refit=True,verbose=2)
	
	# path = os.getcwd() + '/models'
	# os.chdir(path)


	f = open("MyFile.txt", "w")
	f.write('\nSVM: ')
	f.close()
	
	
	kf = KFold(n_splits=10)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		

		test_size = len(X_test)

		f = open("MyFile.txt", "a+")
		f.write('\ntest_size: ')
		f.write(format(test_size))
		f1_after = 0.0
		precision_after = 0.0
		recall_after = 0.0
		support_after = 0.0
		
		

		grid.fit(X_train,y_train)
		# tt = svclassifier.predict_proba(X_test)
		# print(tt)
		
		# fname = 'text_classifier_' + str(test_size)
		# with open(fname, 'wb') as picklefile:
		# 	pickle.dump(grid,picklefile)    

		grid_predictions = grid.predict(X_test)

		

		# print(confusion_matrix(y_test,grid_predictions))
		# print(classification_report(y_test,grid_predictions))

		precision,recall,f1,support=precision_recall_fscore_support(y_test,grid_predictions,average='macro')	
		
		f1_after = f1
		precision_after = precision
		recall_after = recall
		# support_after = support
	
		
		f.write('\nf1-macro: ')
		f.write(format(f1_after))
		f.write('\nprecision-macro: ')
		f.write(format(precision_after))
		f.write('\nrecall-macro: ')
		f.write(format(recall_after))
		# f.write('\nsupport: ')
		# f.write(format(support_after))


		precision,recall,f1,support=precision_recall_fscore_support(y_test,grid_predictions,average='weighted')
		f.write('\nf1-weighted: ')
		f.write(format(f1))
		f.write('\nprecision-weighted: ')
		f.write(format(precision))
		f.write('\nrecall-weighted: ')
		f.write(format(recall))

		# roc = roc_auc_score(y_test, grid_predictions) # print('AUC RF:%.3f'% roc)
		accuracy = accuracy_score(y_test, grid_predictions)
		print(accuracy)

		# f.write('\nAUC: ')
		# f.write(format(roc))
		f.write('\naccuracy: ')
		f.write(format(accuracy))

		f.write('\n')
		f.close()

		   
A = []

# Getting back the objects:
with open('objs.pkl', 'rb') as f:  
    X, y = pickle.load(f)
    # print(X.shape, y.shape)
    # print(type(X))
    for col in X.columns: 
    	A.append(col)

print(len(A))

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot

model = KNeighborsClassifier()

model.fit(X, y)

results = permutation_importance(model, X, y, scoring='accuracy')

importance = results.importances_mean

f = open("about.txt", "w")
f.write('fss:')
f.close()
for i,v in enumerate(importance):
	if (v > 0):
		# print('Feature: %0d, Score: %.5f' % (i,v))		
		print('Score: %.5f' % (v))
		print(A[i])
		f = open("about.txt", "a+")
		f.write('Score: %.5f' % (v))
		f.write(A[i])
		f.close()



	# print(i,' ', v)
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.show()


# test (X, y)










import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def test (X, y):

	tfidfconverter = TfidfTransformer()
	X = tfidfconverter.fit_transform(X).toarray()

	param_grid = {'alpha': np.linspace(0.5, 1.5, 6),'fit_prior': [True, False]}

	
	grid = GridSearchCV(GaussianNB(),param_grid)	



	# classifier = GaussianNB()

	
	# path = os.getcwd() + '/models'
	# os.chdir(path)

	

	f = open("MyFile.txt", "w")
	f.write('\nNB: ')
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
		
		


		# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

		# cross_val_score(model, iris.data, iris.target, cv=10)

		
		grid.fit(X_train,y_train)
		y_pred = grid.predict(X_test)

		# y_pred = classifier.fit(X_train, y_train).predict(X_test)



		# fname = 'text_classifier_' + str(test_size)
		# with open(fname, 'wb') as picklefile:
		# 	pickle.dump(model,picklefile)

		# print(confusion_matrix(y_test,y_pred))
		# print(classification_report(y_test,y_pred))
		# print(accuracy_score(y_test, y_pred))

		precision,recall,f1,support=precision_recall_fscore_support(y_test,y_pred,average='macro')	
		
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


		precision,recall,f1,support=precision_recall_fscore_support(y_test,y_pred,average='weighted')
		f.write('\nf1-weighted: ')
		f.write(format(f1))
		f.write('\nprecision-weighted: ')
		f.write(format(precision))
		f.write('\nrecall-weighted: ')
		f.write(format(recall))

		try:
			roc = roc_auc_score(y_test, y_pred) # print('AUC RF:%.3f'% roc)
			f.write('\nAUC: ')
			f.write(format(roc))
		except:
			pass

		accuracy = accuracy_score(y_test, y_pred)
		print(accuracy)
		f.write('\naccuracy: ')
		f.write(format(accuracy))

		f.write('\n')
		f.close()

		   
		





# Getting back the objects:
with open('objs.pkl', 'rb') as f:  
    X, y = pickle.load(f)


test (X, y)










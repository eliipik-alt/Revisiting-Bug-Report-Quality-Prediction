
from bs4 import BeautifulSoup
import requests
import io 
import nltk
from nltk.corpus import stopwords 
import csv 
import re
import numpy as np
from nltk import bigrams
import itertools
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
import pickle

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer


def reporter_list_func(list_change):

	reporter_list = []

	counter = 0;
	for x in list_change:	
		if (x.find(class_="comment")):
			counter = counter+1
			comment = x.find(class_="comment")
			if(comment.find(class_="layout-table change-head reporter")):
				if(x.find(class_="comment-text")):
					reporter_list.append(x.find(class_="comment-text"))

			else:
				break
	
	# print (type(reporter_list[0])) = <class 'bs4.element.Tag'>

	paragraphs = remove_stop_words(reporter_list)
				
	# print (paragraphs)
	return paragraphs



def remove_stop_words(alist):
	
	stop_words = set(stopwords.words('english')) 

	bad_chars = [';', ':', '!', "*", "(", ")", "[", "]", "{", "}", ".", ",", "``", "''", '""', "?", "/","_",
	"+", "--", "``", "'", "-", ">", "<", "!","@","#","$","%","^","&","*","|","=","`","~","'", 'pre', 'div',
	'<pre','<a>','<p>']

	filtered_string = ''
	filtered_list = [] 
	stem_sentence=''

	for x in alist:
		if(x):
			temp = x.get_text()
		else:
			temp = ''

		token_words=word_tokenize(temp)
		for word in token_words:
			stem_sentence = stem_sentence + lancaster.stem(word) + " "
			

		stem_sentence = re.sub(r'[0-9]+', '', stem_sentence)

		stem_sentence = re.sub(r"\b[a-zA-Z]\b", "", stem_sentence)

		for i in bad_chars: 
			stem_sentence = stem_sentence.replace(i, '')

		
		filtered_string = filtered_string + stem_sentence
		

	filtered_list.append(filtered_string)
	return filtered_list			


def train(rows, columns):

	training_data = pd.DataFrame(rows, columns=columns)



	# Term-Document Matrix
	labels = []
	stmt_docs = []
	for index,row in training_data.iterrows():
		stmt_docs.append(row['sent'])
		labels.append(row['class'])

	vec_s = CountVectorizer()
	X_s = vec_s.fit_transform(stmt_docs)
	tdm_s = pd.DataFrame(X_s.toarray(), columns=vec_s.get_feature_names())
	tdm_s['attachment'] = attachment
	print(tdm_s)
	

	tdm_s['class'] = labels


	X = tdm_s.drop('class', axis=1)
	y = tdm_s['class']

	return X, y




lancaster=LancasterStemmer()

alist_reporter = [] 


ids_labels = {}
attachment = []


with open('data.csv', 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    
    for row in csvreader:
    	if(row[1] == 'actual_label'):
    		continue 
    			
    	if(row[1] == 'bad'):
    		ids_labels[row[0]] = 0	
    	
    	if(row[1] == 'good'):
    		ids_labels[row[0]] = 1
    		
    	if(row[1] == 'neutral'):
    		ids_labels[row[0]] = 2
    		



for id, label in ids_labels.items():

	x = "https://bugzilla.mozilla.org/show_bug.cgi?id=" + id 

	page = requests.get(x)
	# print(page)
	soup = BeautifulSoup(page.content, 'html.parser')
	if (soup.find_all(id="module-attachments-content")):
		attachment.append(1)
	else:
		attachment.append(0)

	# print(attachment)

	list_change = soup.find_all(class_="change-set")

	list_good = reporter_list_func(list_change)
	import readability
	
	results = readability.getmeasures(list_good, lang='en')
	print(results['readability grades']['ARI'])

	list_good.append(label)
	alist_reporter.append(list_good)



columns = ['sent', 'class']

X, y = train (alist_reporter, columns)

# print(X)


def get_text_length(x):
    return np.array(attachment).reshape(-1, 1)
print(get_text_length())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('tfidf', TfidfTransformer()),
        ])),
        ('length', Pipeline([
            ('count', FunctionTransformer(get_text_length, validate=False)),
        ]))
    ])),
    ('clf', OneVsRestClassifier(LinearSVC()))])


classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
print(predicted)


with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([X, y], f)





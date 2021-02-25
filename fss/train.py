
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

		tokens_without_sw=[word for word in token_words if not word in stopwords.words()]

		for word in tokens_without_sw:
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
	tdm_s['class'] = labels


	X = tdm_s.drop('class', axis=1)
	y = tdm_s['class']

	return X, y




lancaster=LancasterStemmer()

alist_reporter = [] 


ids_labels = {}


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
	list_change = soup.find_all(class_="change-set")

	list_good = reporter_list_func(list_change)
	list_good.append(label)
	alist_reporter.append(list_good)



columns = ['sent', 'class']

X, y = train (alist_reporter, columns)


with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([X, y], f)


# from TextFeatureSelection import TextFeatureSelection
# import csv 

# #Multiclass classification problem
# input_doc_list=['i am very happy','i just had an awesome weekend','this is a very difficult terrain to trek. i wish i stayed back at home.','i just had lunch','Do you want chips?']
# target=['Positive','Positive','Negative','Neutral','Neutral']
# fsOBJ=TextFeatureSelection(target=target,input_doc_list=input_doc_list)
# result_df=fsOBJ.getScore()


# dd = result_df.values
# with open('temp.csv', "w") as f:
#     writer = csv.writer(f)
#     for row in dd:
#         writer.writerow(row)

# result_df.to_csv('temp.csv', mode='a', header=False)

# print(result_df.head)



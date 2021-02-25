
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
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator




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
	
	

	paragraphs = remove_stop_words(reporter_list)
	# print(paragraphs)
	# print(len(paragraphs))
	return len(paragraphs)



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
			stem_sentence = lancaster.stem(word)
			

			# stem_sentence = re.sub(r'[0-9]+', '', stem_sentence)

			stem_sentence = re.sub(r"\b[a-zA-Z]\b", "", stem_sentence)

			for i in bad_chars: 
				stem_sentence = stem_sentence.replace(i, '')

			if(stem_sentence != ''):
				filtered_list.append(stem_sentence)

		

	return filtered_list			



def year_count(filename ):
	ids_labels = {}
	counts = []
	years = []

	thisdict = {}

	with open(filename, 'r') as csvfile: 
		csvreader = csv.reader(csvfile) 
	    
		for row in csvreader:
			if(row[1] == 'actual_label'):
				continue
			else:
				ids_labels[row[0]] = row[1]



	for id, label in ids_labels.items():

		x = "https://bugzilla.mozilla.org/show_bug.cgi?id=" + id 

		page = requests.get(x)
		# print(page)
		soup = BeautifulSoup(page.content, 'html.parser')
		list_change = soup.find_all(class_="change-set")

		count = reporter_list_func(list_change)
		print(count)
		list_year = soup.find(class_="rel-time")

		string = list_year.get('data-time')
		year = datetime.utcfromtimestamp(int(string)).strftime('%Y')
		print(year)

		if year in thisdict:
			x, y = thisdict[year]
			thisdict[year] = [x+count, y+1]
			
		else:
			thisdict[year] = [count,1]

	int_thisdict = {int(k) : v for k, v in thisdict.items()}

	newdict = {}
	for key, value in sorted(int_thisdict.items()):
		newdict[key] = value

	print(newdict)
	for year in newdict:
		years.append(year)
		x, y = newdict[year]
		counts.append(x/y)

	return years,counts




lancaster=LancasterStemmer()

years,counts = year_count('data.csv')

rows = zip(years,counts)
with open('year_count.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)


years_cuezilla,counts_cuezilla = year_count('data_cuezilla.csv')

rows = zip(years_cuezilla,counts_cuezilla)
with open('year_count_cuezilla.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)



# plt.plot(years, counts,'r-')


ax = figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.plot(years,counts, label = 'ours')
ax.plot(years_cuezilla,counts_cuezilla, label = 'cuezilla')



# plt.plot(x_axis, y_axis_f1_after,'k-',label = 'f1 after tuning')

plt.legend()
plt.xlabel('year')
plt.ylabel('average length of description')
plt.title('our data vs. cuezilla')
plt.show()













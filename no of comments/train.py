
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

from datetime import datetime

from sklearn import preprocessing

		



ids_labels = {}
ids_labels_bad = {}
ids_labels_good = {}

counts_good = []
counts_bad = []

thisdict = {}




with open('data.csv', 'r') as csvfile: 
	csvreader = csv.reader(csvfile) 
    
	for row in csvreader:
		if(row[1] == 'actual_label'):
			continue
		if(row[1] == 'bad'):
			ids_labels_bad[row[0]] = row[1]	
		else:
			ids_labels_good[row[0]] = row[1]	






for id, label in ids_labels_good.items():

	URL = "https://bugzilla.mozilla.org/rest/bug/" + id + "?include_fields=comment_count"
	
	page = requests.get(URL)
	x = page.json()
	y = x["bugs"]
	
	counts_good.append(y[0]["comment_count"])
	# print(product)

		
for id, label in ids_labels_bad.items():

	URL = "https://bugzilla.mozilla.org/rest/bug/" + id + "?include_fields=comment_count"
	
	page = requests.get(URL)
	x = page.json()
	y = x["bugs"]
	
	counts_bad.append(y[0]["comment_count"])



rows = zip(counts_good, counts_bad)
with open('counts.csv', "w") as f:
	writer = csv.writer(f)
	for row in rows:
		writer.writerow(row)












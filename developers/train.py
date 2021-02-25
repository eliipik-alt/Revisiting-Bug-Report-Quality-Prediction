
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
from requests.auth import HTTPBasicAuth
from datetime import datetime

from sklearn import preprocessing

		


def year_email(filename):
	ids_labels = {}
	emails = []
	years = []
	ids = []

	with open(filename + '.csv', 'r') as csvfile: 
		csvreader = csv.reader(csvfile) 
	    
		for row in csvreader:
			if(row[1] == 'actual_label'):
				continue
			else:
				ids_labels[row[0]] = row[1]



	for id, label in ids_labels.items():

		ids.append(id)

		URL = "https://bugzilla.mozilla.org/rest/bug/" + id + "?include_fields=creation_time,assigned_to,creator"
		
		page = requests.get(URL)
		x = page.json()
		y = x["bugs"]

		year = y[0]["creation_time"]		
		year = year.split('-')[0]
		print(year)
		years.append(year)

		assignee = y[0]["assigned_to"]
		print(assignee)
		emails.append(assignee)
	

	return ids,years,emails


ids,years,emails = year_email('data') # data.csv

rows = zip(ids,years,emails)
with open('year_email.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)





# for login

	# session = requests.Session()

	# # Create the payload
	# payload = {'_username':'[epaikari@uci.edu]', 
	#           '_password':'[3ky7LFhmAm8MGvn]'
	#          }

	# # Post the payload to the site to log in
	# page = session.post("https://bugzilla.mozilla.org/login", data=payload)
	# soup = BeautifulSoup(page.content, 'html.parser')
	# print(soup)
	# login = 'https://bugzilla.mozilla.org/rest/login?login=epaikari@uci.edu&password=3ky7LFhmAm8MGvn'
		# r = requests.get(login)
		# print(r.text)











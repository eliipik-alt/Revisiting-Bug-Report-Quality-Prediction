
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

		


alist_nums = [] 
ids_labels = {}
thisdict = {}
years = []
experience = []



with open('data.csv', 'r') as csvfile: 
	csvreader = csv.reader(csvfile) 
    
	for row in csvreader:
		if(row[1] == 'actual_label'):
			continue
		else:
			ids_labels[row[0]] = row[1]





for id, label in ids_labels.items():

	URL = "https://bugzilla.mozilla.org/rest/bug/" + id + "?include_fields=creator,creation_time"
	
	page = requests.get(URL)
	x = page.json()
	y = x["bugs"]
	
	text = y[0]["creation_time"]
	match = re.search(r'\d{4}-\d{2}-\d{2}', text)
	time = datetime.strptime(match.group(), '%Y-%m-%d').date()
	year = time.strftime('%Y')


	try:
		author = y[0]["creator_detail"]
		email = author["email"]

		
		address = "https://bugzilla.mozilla.org/buglist.cgi?bug_type=defect&classification=Client%20Software&classification=Developer%20Infrastructure&classification=Components&classification=Server%20Software&classification=Other&product=Firefox&product=Firefox%20Build%20System&product=Firefox%20for%20Android&product=Firefox%20for%20Echo%20Show&product=Firefox%20for%20FireTV&product=Firefox%20for%20ios&product=Firefox%20Friends&product=Firefox%20Private%20Network&resolution=FIXED&resolution=INVALID&resolution=WONTFIX&resolution=INACTIVE&resolution=DUPLICATE&resolution=WORKSFORME&resolution=INCOMPLETE&resolution=SUPPORT&resolution=EXPIRED&resolution=MOVED&component=Administration&component=about:logins&component=Activity%20Stream&component=Activity%20Streams:%20General&component=Activity%20Streams:%20Server%20Operations&component=Activity%20Streams:%20Timeline&component=Add-on%20Manager&component=Address%20Bar&component=Android%20partner%20distribution&component=Android%20Studio%20and%20Gradle%20Integration&component=Android%20Sync&component=Audio/Video&component=Awesomescreen&component=Bookmarks%20&%20History&component=Bootstrap%20Configuration&component=Browser&component=Build%20&%20Test&component=Custom%20Tabs&component=Data%20Providers&component=Data%20Storage&component=Developer%20Environment%20Integration&component=Disability%20Access&component=Distributions&component=Download%20Manager&component=Downloads%20Panel&component=Enterprise%20Policies&component=Extension%20Compatibility&component=Favicon%20Handling&component=Favicons&component=File%20Handling&component=Firefox%20Accounts&component=Firefox%20Monitor&component=First%20Run&component=friends.mozilla.org&component=General&component=General:%20Unsupported%20Platforms&component=Generated%20Documentation&component=Headless&component=Home%20screen&component=Installer&component=JimDB&component=Keyboard%20Navigation&component=Keyboards%20and%20IME&component=Launcher%20Process&component=Lint%20and%20Formatting&component=Locale%20switching%20and%20selection&component=Localization&component=Login%20Management&component=Logins&component=Passwords%20and%20Form%20Fill&component=Mach%20Core&component=Menu%20and%20Toolbar&component=Menus&component=Messaging%20System&component=Metrics&component=Migration&component=Mobile&component=New%20Tab%20Page&component=Normandy%20Client&component=Normandy%20Server&component=Other&component=Overlays&component=Page%20Info%20Window&component=PDF%20Viewer&component=Planning&component=Pocket&component=Preferences&component=Private%20Browsing&component=Profile%20Handling&component=Protections%20UI&component=Reader%20View&component=Reading%20List&component=Remote%20Settings%20Client&component=Screencasting&component=Screenshots&component=Search&component=Security&component=Security:%20General&component=Services%20Automation&component=Session%20Restore&component=Settings%20and%20Preferences&component=Shell%20Integration&component=Site%20Identity&component=Site%20Permissions&component=Source%20Code%20Analysis&component=Sync&component=System%20Add-ons:%20Off-train%20Deployment&component=Tabbed%20Browser&component=Task%20Configuration&component=Telemetry&component=Testing&component=Text%20Selection&component=Theme&component=Theme%20&%20Visual%20Design&component=Theme%20and%20Visual%20Design&component=Third%20Party%20Security%20Issues&component=Toolbar&component=Toolbars%20and%20Customization&component=Toolchains&component=Tours&component=Translation&component=Try&component=Untriaged&component=Web%20Apps%20(PWAs)&component=WebPayments%20UI&emailreporter1=1&email1={}&emailtype1=substring&query_format=advanced&f1=creation_ts&o1=lessthan&v1={}".format(email, time)
		
		page = requests.get(address)		
		
		soup = BeautifulSoup(page.content, 'html.parser')

		stat = soup.find(class_="bz_result_count")
		print(stat)
		try:
			num = int(re.search(r'\d+', stat.get_text()).group())
		except:
			num = 0
			pass

		if year in thisdict:
			x, y = thisdict[year]
			x.append(num)
			thisdict[year] = [x, y+1]
		
		else:
			thisdict[year] = [[num],1]

	except:
		pass

int_thisdict = {int(k) : v for k, v in thisdict.items()}
		
newdict = {}
for key, value in sorted(int_thisdict.items()):
	newdict[key] = value

print(newdict)
for year in newdict:
	temp = 0
	years.append(year)
	num, y = newdict[year]
	arr = []
	arr.append(num) # to make it 2D array
	normalized = preprocessing.normalize(arr)
	# print(normalized[0][0])
	for x in normalized[0]:
		temp += x

	score = temp/y
	experience.append(score)
	

	

rows = zip(years,experience)
with open('year_experience.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)


# years_cuezilla,counts_cuezilla = year_experience('data_cuezilla') # data_cuezilla.csv

# rows = zip(years_cuezilla,counts_cuezilla)
# with open('year_experience_cuezilla.csv', "w") as f:
#     writer = csv.writer(f)
#     for row in rows:
#         writer.writerow(row)














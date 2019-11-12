"""
Using NewsAPI:
API Key: 798a9003a6254698ad9117441ce75124
"""
import json
import requests
import csv
import pandas as pd
from newsapi import NewsApiClient
from newspaper import Article

################## Helper Functions ##################
def read_url(url):
	"""
	Read text from url using Newspaper
	"""
	article = Article(url)
	article.download()
	article.parse()
	return article.text

def save_urls(urls, filename):
	"""
	Save url's to csv under given filename
	"""
	rows = [[url] for url in urls]
	with open(filename, 'w', newline='') as out_file:
		wr = csv.writer(out_file)
		wr.writerows(rows)

################## Function ##################

def get_company_urls(company, source_list, save=False):
	"""
	Get URL's for a given company across sources in source_list
	"""
	urls = set()
	api = NewsApiClient(api_key='798a9003a6254698ad9117441ce75124')
	for src in source_list:
		news = api.get_everything(
					q=company,
				    language='en',
				    page_size=10,
				    sort_by='popularity', # Options are popularity, relevancy, and publishedAt
				    sources=src)
		articles = news['articles']
		for a in articles:
			urls.add(a['url'])

	# Save URL's to file
	if save:
		filename = 'urls/' + company + '_urls.csv'
		save_urls(urls, filename)

	return urls

# NOTE: Don't use sources that need subscription. Won't be able to read article.

all_sources = ['abc-news', 'al-jazeera-english', 'ars-technica', 'associated-press',
				'axios', 'bleacher-report', 'bloomberg', 'breitbart-news',
				'business-insider', 'buzzfeed', 'cbs-news', 'cnbc', 'cnn',
				'crypto-coins-news', 'engadget', 'entertainment-weekly', 'espn',
				'espn-cric-info', 'fortune', 'fox-news', 'fox-sports', 'google-news',
				'hacker-news', 'ign', 'mashable', 'medical-news-today', 'msnbc',
				'mtv-news', 'national-geographic', 'national-review', 'nbc-news',
				'new-scientist', 'newsweek', 'new-york-magazine', 'next-big-future',
				'nfl-news', 'nhl-news', 'politico', 'polygon', 'recode', 'reddit-r-all',
				'reuters', 'techcrunch', 'techradar', 'the-american-conservative',
				'the-hill', 'the-huffington-post', 'the-new-york-times', 'the-next-web',
				'the-verge', 'the-wall-street-journal', 'the-washington-post',
				'the-washington-times', 'time', 'usa-today', 'vice-news', 'wired']

chosen_sources = ['cnbc', 'msnbc', 'reuters', 'the-new-york-times', 'the-washington-post', 
					'the-washington-times', 'techcrunch', 'wired', 'the-verge',
					'bloomberg', 'nbc-news']

sp100 = pd.read_csv('sp100.csv')
sp100_companies = sp100['CommonName'].tolist()

first_20 = sp100_companies[:20]
second_20 = sp100_companies[20:40]
third_20 = sp100_companies[40:60]
fourth_20 = sp100_companies[60:80]
fifth_20 = sp100_companies[80:]

# Collect URL's
# First 20 done (11/8/19),
# Second 20 done (11/9/19),
# Third done (11/10/19 @ noon),
# Fourth done (11/11/19 @ midnight),
# Fifth done (11/11/19 @ noon)

for company in fifth_20:
	get_company_urls(company, chosen_sources, save=True)






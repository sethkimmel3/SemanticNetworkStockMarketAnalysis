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

################## Tests ##################
def news_test():
	"""
	Directly call NewsAPI
	"""
	url = ('https://newsapi.org/v2/everything?'
	       'q=Google&'
	       'sortBy=popularity&'
	       'language=en&'
	       'pageSize=100&'
	       'apiKey=798a9003a6254698ad9117441ce75124')

	r = requests.get(url)
	data = r.json()
	print(data['articles'][0]['url'])
	for article in data['articles']:
		print(article['url'])
	# with open('data.json', 'w') as f:
		# json.dump(data,f)

def news_api_test():
	"""
	Call with NewsAPI Python library
	"""
	api = NewsApiClient(api_key='798a9003a6254698ad9117441ce75124')
	out = api.get_everything(
		q='Microsoft',
	    language='en',
	    page_size=100
	    )

	sources = api.get_sources(language="en", country="us")
	print([src['id'] for src in sources['sources']])

def test_multiple_sources():
	api = NewsApiClient(api_key='798a9003a6254698ad9117441ce75124')
	for src in very_small:
		print(src)
		out = api.get_everything(
			q='Microsoft',
		    language='en',
		    page_size=10,
		    sources=src
		    )
		for article in out['articles']:
			print(article['title'])

################## Helper Functions ##################
def read_url(url):
	"""
	Read text from url using Newspaper
	"""
	article = Article(url)
	article.download()
	article.parse()
	return article.text

def get_all_urls(companies, source_list):
	urls = set()
	for comp in companies:
		cur = get_company_urls(comp, source_list)
		urls = urls.union(cur)

	return urls

def count_mentions(companies, urls):
	zero = np.zeros((len(companies),len(companies)))
	counts = pd.DataFrame(zero,index=companies, columns=companies)

	count = 1
	for url in urls:
		if count % 10 == 0:
			print('Reading URL %d of %d' % (count, len(urls)))

		try:
			article = read_url(url).lower()
		except:
			article = ''
		# Check which companies are in the article
		in_article = [comp for comp in companies if comp.lower() in article]
		for comp1 in in_article:
			for comp2 in in_article:
				if comp1 != comp2:
					counts[comp1][comp2] += 1

		count += 1

	return counts

def save_urls(urls, filename):
	"""
	Save url's to csv under given filename
	"""
	rows = [[url] for url in urls]
	with open(filename, 'w', newline='') as out_file:
		wr = csv.writer(out_file)
		wr.writerows(rows)

################## Functions ##################

def run(companies, sources):
	print('Getting URLs...')
	all_urls = get_all_urls(companies, sources)
	print('Counting...')
	counts = count_mentions(companies, all_urls)
	
	return counts

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

"""
Note: So rn looks like number of requests is going to be (#companies) x (#sources)
Ex: For 100 companies and like 25 sources, that's 2500 requests
NewsAPI limits a user to 500 requests per day (250 per 12 hours)
So each user can do ~20 companies per day (10 per 12 hours)
"""

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

small_sources = ['abc-news', 'bloomberg', 'cbs-news', 'cnn', 'nbc-news',
				'politico', 'reddit-r-all','reuters', 'techcrunch', 'techradar',
				'the-huffington-post', 'the-new-york-times',
				'the-wall-street-journal', 'the-washington-post','time', 'usa-today']

chosen_sources = ['cnbc', 'msnbc', 'reuters', 'the-new-york-times', 'the-washington-post', 
					'the-washington-times', 'techcrunch', 'wired', 'the-verge',
					'bloomberg', 'nbc-news']

companies = ['google', 'facebook', 'amazon', 'apple', 'nvidia', 'uber', 'exxon', 'netflix', 'disney', 'verizon', 'chipotle']
small_companies = ['google', 'facebook', 'apple', 'biogen']
sp100 = pd.read_csv('sp100.csv')

sp100_companies = sp100['CommonName'].tolist()
first_20 = sp100_companies[:20]
second_20 = sp100_companies[20:40]
third_20 = sp100_companies[40:60]

# print(run(sp100_companies, ['cnbc'])) # Test to just see if this works w all the S&P 100 companies

# Collect URL's
# NOTE: First 20 done (11/8/19), second 20 done (11/9/19), third done (11/10/19 @ noon)
for company in third_20:
	get_company_urls(company, chosen_sources, save=True)






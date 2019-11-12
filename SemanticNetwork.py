"""
Build the network given a list of URL's
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
from newspaper import Article

class SemanticNetwork:
    def __init__(self, adj_path, sectors_path):
        # If given path to CSV representation of adjacency matrix, just load it in
        if os.path.isfile(adj_path):
            adj = pd.read_csv(adj_path, index_col=0)
        # If given path to folder containing CSV's of URL's, create adjacency matrix from these files
        elif os.path.isdir(adj_path):
            companies, urls = load_urls('adj_path')
            adj = count_mentions(companies, urls)
            # Save to CSV for future use
            adj.to_csv(adj_path + '.csv')
        else:
            raise ValueError('Input should be path to adjacency matrix CSV or folder of CSVs of URLs')
        
        self.adj = adj
        self.graph = nx.from_pandas_adjacency(adj)

        # Add sector as attribute
        sector_df = pd.read_csv(sectors_path)
        sector_map = {row['CommonName']:row['Sector'] for __, row in sector_df.iterrows()}
        # sector_map = sector_df['CommonName'].map(sector_df['Sector'])
        nx.set_node_attributes(self.graph, sector_map, name='sector')        

    def load_urls(urls_path):
        """
        Use this if given path to folder of URL's
        Input: Path to folder containing CSV files that contain URL's
        Output: List of companies and set of all URL's
        """
        companies = []
        # Use a set to avoid repeated articles
        urls = set()
        # Iterate over URL's in the folder
        for filename in os.listdir(urls_path):
            # Ignore hidden files
            if not filename.startswith('.'):
                # Extract company name from csv file name
                company = filename.split('_')[0]
                companies.append(company)
                file_path = urls_path + '/' + filename
                # Read CSV to get URL's
                with open(file_path) as f:
                    for url in f:
                        urls.add(url.strip())

        return companies, urls


    def read_url(url):
        """
        Read text from url using Newspaper API
        Returns article content as one continuous string
        """
        article = Article(url)
        article.download()
        article.parse()
        return article.text

    def count_mentions(companies, urls):
        """
        Create adjacency matrix of companies given set of URL's to articles
        """
        zero = np.zeros((len(companies),len(companies)))
        counts = pd.DataFrame(zero,index=companies, columns=companies)

        count = 1
        print('Reading URLs...')
        for url in urls:
            if count % 100 == 0:
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


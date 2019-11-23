"""
Draw Inferences and Correlations on the Network and Financial Time-Series Data
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from SemanticNetwork import SemanticNetwork
import json

print('Loading graph...')
network = SemanticNetwork('adj_matrix.csv', 'sp100.csv')
G = network.graph

"""
Analyses on the network alone
"""

sp100 = pd.read_csv('sp100.csv')
sp100_symbols = sp100['Symbol'].tolist()
sp100_names = sp100['Name'].tolist()
sp100_common_names= sp100['CommonName'].tolist()
sp100_sectors = sp100['Sector'].tolist()

# map symbols to common names
sp100_symbols_to_common_names = {}
for i in range(len(sp100_symbols)):
    sp100_symbols_to_common_names[sp100_symbols[i]] = sp100_common_names[i]


# Finding economically important stocks
def economic_importance_analysis():
    # get the market cap data
    with open('market_cap_data_11_20_19.json') as json_file:
        market_caps = json.load(json_file)

    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G,weight='weight')
    market_cap_and_betweenness = {}

    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    market_cap_and_degree_centrality = {}

    # Eigenvector centrality
    eig_centrality = nx.eigenvector_centrality(G,weight='weight')
    market_cap_and_eig_centrality = {}

    for k,v in market_caps.items():
        market_cap_and_betweenness[k] = (v, betweenness[sp100_symbols_to_common_names[k]])
        market_cap_and_degree_centrality[k] = (v, degree_centrality[sp100_symbols_to_common_names[k]])
        market_cap_and_eig_centrality[k] = (v, eig_centrality[sp100_symbols_to_common_names[k]])

    # plot log of market cap vs. centralities
    for type in ['Betweenness', 'Degree', 'Eigenvector']:
        # get x, y data points
        if(type == 'Betweenness'):
            x = np.asarray([v[0] for k,v in market_cap_and_betweenness.items()], dtype=np.float)
            y = np.asarray([v[1] for k,v in market_cap_and_betweenness.items()], dtype=np.float)
        elif(type == 'Degree'):
            x = np.asarray([v[0] for k, v in market_cap_and_degree_centrality.items()], dtype=np.float)
            y = np.asarray([v[1] for k, v in market_cap_and_degree_centrality.items()], dtype=np.float)
        elif(type == 'Eigenvector'):
            x = np.asarray([v[0] for k, v in market_cap_and_eig_centrality.items()], dtype=np.float)
            y = np.asarray([v[1] for k, v in market_cap_and_eig_centrality.items()], dtype=np.float)

        # plot it
        plt.scatter(x,y)
        plt.title('Market Cap vs.' + type + ' Centrality')
        plt.xlabel('Market Cap')
        plt.ylabel(type + ' Centrality')
        plt.show()


# create new indexes from communities
def create_indexes(num_to_generate):
    row_labels,column_labels = network.adj.index.values,network.adj.columns.values
    max_value, min_value = np.max(network.adj.values), np.min(network.adj.values)
    rows,columns = network.adj.shape

    num_bins = num_to_generate
    bin_values = np.linspace(min_value, max_value, num_bins)
    min_bins = {}
    for i in range(num_bins):
        min_bins[i] = []

    # if two companies have an edge weight above each bin value, put them in the index for that bin
    # this is one way of going about it - not all companies in an index will mutually have that minimum weight between them
    # but all companies will have a pair where that is true
    for i in range(rows):
        for j in range(columns):
            if(i != j):
                for k in range(num_bins):
                    row_label, col_label = row_labels[i], column_labels[j]
                    if(bin_values[k] <= network.adj.iat[i,j]):
                        if(row_label not in min_bins[k]):
                            min_bins[k].append(row_label)
                        if(col_label not in min_bins[k]):
                            min_bins[k].append(col_label)

    for k in range(num_bins):
        print(min_bins[k])

    # let's try the inverse problem - add all companies to every index, and throw them out if they have any weighting above the bins threshold
    max_bins = {}
    for i in range(num_bins):
        max_bins[i] = list(row_labels)

    for i in range(rows):
        for j in range(columns):
            if(i != j):
                for k in range(num_bins):
                    row_label, col_label = row_labels[i], column_labels[j]
                    if(bin_values[k] <= network.adj.iat[i,j]):
                        if(row_label in max_bins[k]):
                            max_bins[k].remove(row_label)
                        if(col_label in max_bins[k]):
                            max_bins[k].remove(col_label)

    for k in range(num_bins):
        print(max_bins[k])

#economic_importance_analysis()

min_indexes, max_indexes = create_indexes(num_to_generate=10)






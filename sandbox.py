"""
Sandbox for testing
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from SemanticNetwork import SemanticNetwork
from itertools import count

print('Loading graph...')
network = SemanticNetwork('adj_matrix.csv', 'sp100.csv')
G = network.graph

print('\n################## Connected Components ##################')
# Connected components
# One large component w/93 companies
# 7 remaining companies all disconnected
components = sorted(nx.connected_components(G), key=len, reverse=True)
print('Sizes of connected components: ', [len(c) for c in components])
print('Disconnected nodes: ', components[1:])

# Compute and plot degree distribution
print('\n################## Degree ##################')
degree_sequence = [d for n, d in G.degree()]  # degree sequence
plt.figure(1)
plt.hist(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 10, 10), color='b')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")

avg_degree = np.mean(degree_sequence)
print('Average Degree: ', avg_degree)

# Which companies have highest degree?
print('Companies with highest degree:')
degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
for i in range(10):
    print(degrees[i])

# Compute and plot local clustering coefficients
print('\n################## Local Clustering Coefficient ##################')
clustering = nx.clustering(G,weight='weight')
clustering_sequence = list(clustering.values())
plt.figure(2)
plt.hist(clustering_sequence, bins=np.arange(min(clustering_sequence), max(clustering_sequence) + 0.001, 0.001), color='b')
plt.title("Local Clustering Coefficient Histogram")
plt.ylabel("Count")
plt.xlabel("Local Clustering Coefficient")

avg_clustering = np.mean(clustering_sequence)
print('Average Local Clustering Coefficient: ', avg_clustering)

# Which companies have highest clustering coefficients?
print('Companies with highest clustering coefficient:')
clust = sorted(clustering.items(), key=lambda x: x[1], reverse=True)
for i in range(10):
    print(clust[i])


# Betweenness centrality
print('\n################## Betweenness Centrality ##################')
betweenness = nx.betweenness_centrality(G,weight='weight')
betweenness_sequence = list(betweenness.values())
plt.figure(3)
plt.hist(betweenness_sequence, bins=np.arange(min(betweenness_sequence), max(betweenness_sequence) + 0.001, 0.001), color='b')
plt.title("Betweenness Centrality Histogram")
plt.ylabel("Count")
plt.xlabel("Betweenness Centrality")

avg_betweenness = np.mean(betweenness_sequence)
print('Average Betweenness Centrality: ', avg_betweenness)

# Which companies have highest betweenness centrality?
print('Companies with highest betweenness centrality:')
bet = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
for i in range(10):
    print(bet[i])

# Degree centrality
print('\n################## Degree Centrality ##################')
degree_centrality = nx.degree_centrality(G)
degree_centrality_sequence = list(degree_centrality.values())
plt.figure(3)
plt.hist(degree_centrality_sequence, bins=np.arange(min(degree_centrality_sequence), max(degree_centrality_sequence) + 0.001, 0.001), color='b')
plt.title("Degree Centrality Histogram")
plt.ylabel("Count")
plt.xlabel("Degree Centrality")

avg_degree_centrality = np.mean(degree_centrality_sequence)
print('Average Degree Centrality: ', avg_degree_centrality)

# Which companies have highest degree centrality?
print('Companies with highest degree centrality:')
deg = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
for i in range(10):
    print(deg[i])

# Eigenvector centrality
print('\n################## Eigenvector Centrality ##################')
eig_centrality = nx.eigenvector_centrality(G,weight='weight')
eig_centrality_sequence = list(eig_centrality.values())
plt.figure(3)
plt.hist(eig_centrality_sequence, bins=np.arange(min(eig_centrality_sequence), max(eig_centrality_sequence) + 0.001, 0.001), color='b')
plt.title("Eigenvector Centrality Histogram")
plt.ylabel("Count")
plt.xlabel("Eigenvector Centrality")

avg_eig_centrality = np.mean(eig_centrality_sequence)
print('Average Eigenvector Centrality: ', avg_eig_centrality)

# Which companies have highest eigenvector centrality?
print('Companies with highest eigenvector centrality:')
eig = sorted(eig_centrality.items(), key=lambda x: x[1], reverse=True)
for i in range(10):
    print(eig[i])


# Map node classes to colors and use to plot graph
color_mapping = {'Communication Services':'#15F5F0',
				 'Consumer Discretionary':'#FA3D35',
				 'Consumer Staples':'#F99B37',
				 'Energy':'#FAF574',
				 'Financials':'#88FA74',
				 'Health Care':'#FBB5E3',
				 'Industrials':'#D3AAFB',
				 'Information Technology':'#3392FF',
				 'Materials':'#FC40FA',
				 'Real Estate':'#859FD5',
				 'Utilities':'#CACBC3'}
colors = [color_mapping[nx.get_node_attributes(G,'sector')[n]] for n in G.nodes()]
sizes = [x+10 for x in degree_sequence]
plt.figure(4)
plt.title('Semantic Network')
nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True, node_size=sizes, node_color=colors, alpha=0.9, width=0.1, font_size=6, font_color='black', font_weight='bold')

# What if we throw out small edges?
edgelist = list(G.edges.data('weight', default=0))
edgelist.sort(key=lambda x:x[2], reverse=True)
small_edgelist = [x for x in edgelist if x[2] >= 10]

plt.figure(5)
plt.title('Semantic Network: Edge Weight >= 10')
nx.draw_networkx(G, edgelist=small_edgelist, with_labels=True, node_size=sizes, node_color=colors, alpha=0.9, width=0.1, font_size=6, font_color='black', font_weight='bold')
plt.show()

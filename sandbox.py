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

network = SemanticNetwork('adj_matrix.csv', 'sp100.csv')
G = network.graph

# Compute and plot degree distribution
degree_sequence = [d for n, d in G.degree()]  # degree sequence
plt.figure(1)
plt.hist(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 10, 10), color='b')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")

avg_degree = np.mean(degree_sequence)
print('Average Degree: ', avg_degree)


# Compute and plot local clustering coefficients
clustering = nx.clustering(G,weight='weight')
clustering_sequence = list(clustering.values())
plt.figure(2)
plt.hist(clustering_sequence, bins=np.arange(min(clustering_sequence), max(clustering_sequence) + 0.001, 0.001), color='b')
plt.title("Local Clustering Coefficient Histogram")
plt.ylabel("Count")
plt.xlabel("Local Clustering Coefficient")

avg_clustering = np.mean(clustering_sequence)
print('Average Local Clustering Coefficient: ', avg_clustering)


# Betweenness centrality
betweenness = nx.betweenness_centrality(G,weight='weight')
betweenness_sequence = list(betweenness.values())
plt.figure(3)
plt.hist(betweenness_sequence, bins=np.arange(min(betweenness_sequence), max(betweenness_sequence) + 0.001, 0.001), color='b')
plt.title("Betweenness Centrality Histogram")
plt.ylabel("Count")
plt.xlabel("Betweenness Centrality")

avg_betweenness = np.mean(betweenness_sequence)
print('Average Betweenness Centrality: ', avg_betweenness)

# Degree centrality
degree_centrality = nx.degree_centrality(G)
degree_centrality_sequence = list(degree_centrality.values())
plt.figure(3)
plt.hist(degree_centrality_sequence, bins=np.arange(min(degree_centrality_sequence), max(degree_centrality_sequence) + 0.001, 0.001), color='b')
plt.title("Degree Centrality Histogram")
plt.ylabel("Count")
plt.xlabel("Degree Centrality")

avg_degree_centrality = np.mean(degree_centrality_sequence)
print('Average Degree Centrality: ', avg_degree_centrality)

# Eigenvector centrality
eig_centrality = nx.eigenvector_centrality(G,weight='weight')
eig_centrality_sequence = list(eig_centrality.values())
plt.figure(3)
plt.hist(eig_centrality_sequence, bins=np.arange(min(eig_centrality_sequence), max(eig_centrality_sequence) + 0.001, 0.001), color='b')
plt.title("Eigenvector Centrality Histogram")
plt.ylabel("Count")
plt.xlabel("Eigenvector Centrality")

avg_eig_centrality = np.mean(eig_centrality_sequence)
print('Average Eigenvector Centrality: ', avg_eig_centrality)


# Map node classes to colors and use to plot graph
classes = set(nx.get_node_attributes(G,'sector').values())
mapping = dict(zip(sorted(classes),count()))
nodes = G.nodes()
colors = [mapping[nx.get_node_attributes(G,'sector')[n]] for n in nodes]
plt.figure(4)
nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True, node_size=degree_sequence, node_color=colors, alpha=0.9, width=0.1, font_size=6, font_color='black')
# ax = plt.gca()
# ax.set_facecolor('black')
plt.show()

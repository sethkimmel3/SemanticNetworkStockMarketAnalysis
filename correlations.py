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
import operator

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
sp100_common_names_to_symbols = {}
for i in range(len(sp100_symbols)):
    sp100_symbols_to_common_names[sp100_symbols[i]] = sp100_common_names[i]
    sp100_common_names_to_symbols[sp100_common_names[i]] = sp100_symbols[i]


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

    # for k in range(num_bins):
    #     print(min_bins[k])

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

    # for k in range(num_bins):
    #     print(max_bins[k])

    return min_bins,max_bins

"""
Correlations to prices - node measures
"""
# load the time-series data
with open('time_series_data_from=2019-07-05to=2019-11-08.json') as json_file:
    price_data = json.load(json_file)



# map symbols to closing prices
symbol_to_closing_price = {}
for symbol,price_info in price_data.items():
    closing_prices = np.zeros(90)
    count = 0
    for k,v in price_info.items():
        closing_prices[count] = v['close']
        count += 1
    symbol_to_closing_price[symbol] = closing_prices


# correlate centrality with past price movements of neighbors
def centrality_to_neighborhood_prices(past_or_future, n_timesteps, k_timesteps, weighted=False):
    # turn the time-series into price changes according to n_timesteps
    symbol_to_price_changes = {}
    for symbol in sp100_symbols:
        price_changes = np.zeros(90 - n_timesteps)
        for i in range(len(symbol_to_closing_price[symbol])):
            if(i > n_timesteps):
                price_change = (symbol_to_closing_price[symbol][i] - symbol_to_closing_price[symbol][i - n_timesteps])/symbol_to_closing_price[symbol][i - n_timesteps]
                price_changes[i - n_timesteps] = price_change
        symbol_to_price_changes[symbol] = price_changes

    neighborhood_average_price_changes = {}
    diff_symbol_and_neighborhood_price_changes = {}
    for symbol in sp100_symbols:
        # get the neighbors
        neighbors = G.neighbors(sp100_symbols_to_common_names[symbol])
        neighbor_symbols = [sp100_common_names_to_symbols[common_name] for common_name in neighbors]
        # get average price change of neighbors for each timestep
        neighborhood_average_price_changes[symbol] = np.zeros(90 - n_timesteps)
        total_edge_weights = 0
        for neighbor in neighbor_symbols:
            if(weighted == True):
                for i in range(90 - n_timesteps):
                    neighborhood_average_price_changes[symbol][i] += network.adj.at[sp100_symbols_to_common_names[symbol],sp100_symbols_to_common_names[neighbor]]*symbol_to_price_changes[neighbor][i]
                total_edge_weights += network.adj.at[sp100_symbols_to_common_names[symbol],sp100_symbols_to_common_names[neighbor]]
            else:
                for i in range(90 - n_timesteps):
                    neighborhood_average_price_changes[symbol][i] += symbol_to_price_changes[neighbor][i]
        if(len(neighbor_symbols) != 0):
            if(weighted == True):
                neighborhood_average_price_changes[symbol] = neighborhood_average_price_changes[symbol] /total_edge_weights
            else:
                neighborhood_average_price_changes[symbol] = neighborhood_average_price_changes[symbol]/len(neighbor_symbols)

        # take the difference between each symbols price changes with its neighborhood
        if(past_or_future == 'past'):
            diff_symbol_and_neighborhood_price_changes[symbol] = np.zeros(90 - n_timesteps)
            for i in range(90 - n_timesteps):
                diff_symbol_and_neighborhood_price_changes[symbol][i] = abs(symbol_to_price_changes[symbol][i] - neighborhood_average_price_changes[symbol][i])
        elif(past_or_future == 'future'):
            diff_symbol_and_neighborhood_price_changes[symbol] = np.zeros(90 - n_timesteps - k_timesteps)
            for i in range(90 - n_timesteps - k_timesteps):
                diff_symbol_and_neighborhood_price_changes[symbol][i] = abs(symbol_to_price_changes[symbol][i] - neighborhood_average_price_changes[symbol][i+k_timesteps])

        # sum the difference vector for each symbol - this will represent our correlations
        diff_symbol_and_neighborhood_price_changes[symbol] = np.sum(diff_symbol_and_neighborhood_price_changes[symbol])


    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight='weight')
    diff_neighborhood_price_changes_and_betweenness = {}

    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    diff_neighborhood_price_changes_and_degree_centrality = {}

    # Eigenvector centrality
    eig_centrality = nx.eigenvector_centrality(G, weight='weight')
    diff_neighborhood_price_changes_and_eig_centrality = {}

    for k, v in diff_symbol_and_neighborhood_price_changes.items():
        diff_neighborhood_price_changes_and_betweenness[k] = (v, betweenness[sp100_symbols_to_common_names[k]])
        diff_neighborhood_price_changes_and_degree_centrality[k] = (v, degree_centrality[sp100_symbols_to_common_names[k]])
        diff_neighborhood_price_changes_and_eig_centrality[k] = (v, eig_centrality[sp100_symbols_to_common_names[k]])

    # plot log of market cap vs. centralities
    for type in ['Betweenness', 'Degree', 'Eigenvector']:
        # get x, y data points
        if (type == 'Betweenness'):
            x = np.asarray([v[0] for k, v in diff_neighborhood_price_changes_and_betweenness.items()], dtype=np.float)
            y = np.asarray([v[1] for k, v in diff_neighborhood_price_changes_and_betweenness.items()], dtype=np.float)
        elif (type == 'Degree'):
            x = np.asarray([v[0] for k, v in diff_neighborhood_price_changes_and_degree_centrality.items()], dtype=np.float)
            y = np.asarray([v[1] for k, v in diff_neighborhood_price_changes_and_degree_centrality.items()], dtype=np.float)
        elif (type == 'Eigenvector'):
            x = np.asarray([v[0] for k, v in diff_neighborhood_price_changes_and_eig_centrality.items()], dtype=np.float)
            y = np.asarray([v[1] for k, v in diff_neighborhood_price_changes_and_eig_centrality.items()], dtype=np.float)

        # plot it
        plt.scatter(x, y)
        plt.title('Absolute Difference in Node and Neighborhood \n Price Changes (' + past_or_future + ' values) vs. ' + type + ' Centrality')
        plt.xlabel('Absolute Difference in Node and Neighborhood \n Price Changes (' + past_or_future + ' values)')
        plt.ylabel(type + ' Centrality')
        plt.show()

"""
Correlations to prices - pair/edge measures
"""

# correlate price change differences to degree separation
def degree_separation_to_price_changes(past_or_future, n_timesteps, k_timesteps):
    # turn the time-series into price changes according to n_timesteps
    symbol_to_price_changes = {}
    for symbol in sp100_symbols:
        price_changes = np.zeros(90 - n_timesteps)
        for i in range(len(symbol_to_closing_price[symbol])):
            if (i > n_timesteps):
                price_change = (symbol_to_closing_price[symbol][i] - symbol_to_closing_price[symbol][i - n_timesteps])/symbol_to_closing_price[symbol][i - n_timesteps]
                price_changes[i - n_timesteps] = price_change
        symbol_to_price_changes[symbol] = price_changes

    distance_to_price_changes = {}
    for node1 in G.nodes():
        for node2 in G.nodes():
            if(nx.has_path(G, node1, node2) and node1 != node2):
                shortest_path = nx.dijkstra_path(G, node1, node2, weight=None)
                path_total = len(shortest_path) - 1

                # get price change differences
                if (past_or_future == 'past'):
                    diff_price_changes = np.zeros(90 - n_timesteps)
                    for i in range(90 - n_timesteps):
                        diff_price_changes[i] = abs(symbol_to_price_changes[sp100_common_names_to_symbols[node1]][i] - symbol_to_price_changes[sp100_common_names_to_symbols[node2]][i])
                elif (past_or_future == 'future'):
                    diff_price_changes = np.zeros(90 - n_timesteps - k_timesteps)
                    for i in range(90 - n_timesteps - k_timesteps):
                        diff_price_changes[i] = abs(symbol_to_price_changes[sp100_common_names_to_symbols[node1]][i] - symbol_to_price_changes[sp100_common_names_to_symbols[node2]][i + k_timesteps])

                diff_price_changes = np.sum(diff_price_changes)

                if(path_total in distance_to_price_changes):
                    distance_to_price_changes[path_total].append(diff_price_changes)
                else:
                    distance_to_price_changes[path_total] = [diff_price_changes]

    for k,v in distance_to_price_changes.items():
        distance_to_price_changes[k] = np.average(v)

    for k,v in distance_to_price_changes.items():
        print(str(k) + ' : ' + str(v))

# correlate price changes differences and weight separation (adjacent only)
def weight_separation_to_price_changes(past_or_future, n_timesteps, k_timesteps):
    # turn the time-series into price changes according to n_timesteps
    symbol_to_price_changes = {}
    for symbol in sp100_symbols:
        price_changes = np.zeros(90 - n_timesteps)
        for i in range(len(symbol_to_closing_price[symbol])):
            if (i > n_timesteps):
                price_change = (symbol_to_closing_price[symbol][i] - symbol_to_closing_price[symbol][i - n_timesteps]) / \
                               symbol_to_closing_price[symbol][i - n_timesteps]
                price_changes[i - n_timesteps] = price_change
        symbol_to_price_changes[symbol] = price_changes

    computed = []
    weight_to_price_changes = {}
    for node1 in G.nodes():
        for node2 in G.nodes():
            if(G.has_edge(node1, node2) and node1 != node2 and (node1,node2) not in computed and (node2,node1) not in computed):
                # get weight separation
                weight_separation = network.adj.at[node1, node2]

                # get price change differences
                if (past_or_future == 'past'):
                    diff_price_changes = np.zeros(90 - n_timesteps)
                    for i in range(90 - n_timesteps):
                        diff_price_changes[i] = abs(symbol_to_price_changes[sp100_common_names_to_symbols[node1]][i] - symbol_to_price_changes[sp100_common_names_to_symbols[node2]][i])
                elif (past_or_future == 'future'):
                    diff_price_changes = np.zeros(90 - n_timesteps - k_timesteps)
                    for i in range(90 - n_timesteps - k_timesteps):
                        diff_price_changes[i] = abs(symbol_to_price_changes[sp100_common_names_to_symbols[node1]][i] - symbol_to_price_changes[sp100_common_names_to_symbols[node2]][i + k_timesteps])

                diff_price_changes = np.sum(diff_price_changes)

                if (weight_separation in weight_to_price_changes):
                    weight_to_price_changes[weight_separation].append(diff_price_changes)
                else:
                    weight_to_price_changes[weight_separation] = [diff_price_changes]

                computed.append((node1,node2))

    for k, v in weight_to_price_changes.items():
        weight_to_price_changes[k] = np.average(v)

    sorted_changes = sorted(weight_to_price_changes.items(), key=operator.itemgetter(0))

    x = np.asarray([item[0] for item in sorted_changes], dtype=np.float)
    y = np.asarray([item[1] for item in sorted_changes], dtype=np.float)

    plt.scatter(x, y)
    plt.title('Absolute Difference in Node to Node \n Price Changes (' + past_or_future + ' values) vs. Weight Separation')
    plt.xlabel('Weight Separation')
    plt.ylabel('Absolute Difference in Node to Node \n Price Changes (' + past_or_future + ' values)')
    plt.show()

"""
Correlations to prices - communities/cliques
"""

# correlate the generated indexes to price change differences
def correlate_strength_of_indexes_with_price_change_differences(indexes, past_or_future, n_timesteps, k_timesteps):
    # turn the time-series into price changes according to n_timesteps
    symbol_to_price_changes = {}
    for symbol in sp100_symbols:
        price_changes = np.zeros(90 - n_timesteps)
        for i in range(len(symbol_to_closing_price[symbol])):
            if (i > n_timesteps):
                price_change = (symbol_to_closing_price[symbol][i] - symbol_to_closing_price[symbol][i - n_timesteps]) / \
                               symbol_to_closing_price[symbol][i - n_timesteps]
                price_changes[i - n_timesteps] = price_change
        symbol_to_price_changes[symbol] = price_changes

    # we're going to get the average strength of the associations in the index and correlate that with price change differences
    average_strength_of_index_to_price_change_differences = []
    for k,companies in indexes.items():
        if(len(companies) != 0):
            weights_in_index = []
            price_change_differences_in_index = []
            computed = []
            for company1 in companies:
                for company2 in companies:
                    if(company1 != company2 and (company1, company2) not in computed and (company2, company1) not in computed):
                        # get the weight between the two companies
                        weights_in_index.append(network.adj.at[company1,company2])

                        # get price change differences
                        if (past_or_future == 'past'):
                            diff_price_changes = np.zeros(90 - n_timesteps)
                            for i in range(90 - n_timesteps):
                                diff_price_changes[i] = abs(symbol_to_price_changes[sp100_common_names_to_symbols[company1]][i] - symbol_to_price_changes[sp100_common_names_to_symbols[company2]][i])
                        elif (past_or_future == 'future'):
                            diff_price_changes = np.zeros(90 - n_timesteps - k_timesteps)
                            for i in range(90 - n_timesteps - k_timesteps):
                                diff_price_changes[i] = abs(symbol_to_price_changes[sp100_common_names_to_symbols[company1]][i] - symbol_to_price_changes[sp100_common_names_to_symbols[company2]][i + k_timesteps])

                        diff_price_changes = np.sum(diff_price_changes)
                        price_change_differences_in_index.append(diff_price_changes)

            average_weights_in_index = np.average(weights_in_index)
            average_price_change_differences_in_index = np.average(price_change_differences_in_index)
            average_strength_of_index_to_price_change_differences.append((average_weights_in_index, average_price_change_differences_in_index))

    x = np.asarray([item[0] for item in average_strength_of_index_to_price_change_differences], dtype=np.float)
    y = np.asarray([item[1] for item in average_strength_of_index_to_price_change_differences], dtype=np.float)

    plt.scatter(x, y)
    plt.title('Absolute Difference in Community \n Price Changes (' + past_or_future + ' values) vs. Average Weight of Community')
    plt.xlabel('Average Weight of Community')
    plt.ylabel('Absolute Difference in Community \n Price Changes (' + past_or_future + ' values)')
    plt.show()

"""
Run the Individual Analyses Here
"""

economic_importance_analysis()

min_indexes, max_indexes = create_indexes(num_to_generate=10)

# for the following analyses, you can vary n and k in loops to see the impact

centrality_to_neighborhood_prices('past',n_timesteps=3, k_timesteps=0, weighted=True)
centrality_to_neighborhood_prices('future',n_timesteps=3, k_timesteps=1, weighted=True)

degree_separation_to_price_changes('past', n_timesteps=1, k_timesteps=0)
degree_separation_to_price_changes('future', n_timesteps=3, k_timesteps=1)

weight_separation_to_price_changes('past', n_timesteps=1, k_timesteps=0)
weight_separation_to_price_changes('future', n_timesteps=3, k_timesteps=1)

correlate_strength_of_indexes_with_price_change_differences(min_indexes, 'past', n_timesteps=3, k_timesteps=0)
correlate_strength_of_indexes_with_price_change_differences(min_indexes, 'future', n_timesteps=3, k_timesteps=1)

correlate_strength_of_indexes_with_price_change_differences(max_indexes, 'past', n_timesteps=3, k_timesteps=0)
correlate_strength_of_indexes_with_price_change_differences(max_indexes, 'future', n_timesteps=3, k_timesteps=1)
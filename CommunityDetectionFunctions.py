from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.cm as cm
import sklearn.metrics.cluster as sk_metrics

class CommunityDetectionFunctions:

    kamada_kawai_pos = None

    #-----------------------------------------------------------------------------------------------
    # Evaluation functions
    #-----------------------------------------------------------------------------------------------
    def compute_jaccard_index(self, communities1, communities2):
        agreements = 0
        dissagreements = 0
        
        community_pairs = [pair for pair in itertools.product(communities1, communities2) if len(set(pair[0]).intersection(set(pair[1]))) >= 2]
        node_pairs = [[pair for pair in itertools.combinations(set(partition_i).union(set(partition_j)), 2)] for partition_i, partition_j in community_pairs]
        pairs = list(set([item for sublist in node_pairs for item in sublist]))
    
        for pair in pairs:
            exist_i = any(pair[0] in partition_i and pair[1] in partition_i for partition_i in communities1)
            exist_j = any(pair[0] in partition_j and pair[1] in partition_j for partition_j in communities2)
            if exist_i and exist_j:
                agreements += 1
            elif exist_i or exist_j:
                dissagreements += 1
        return agreements / (agreements + dissagreements)
    
    def compute_normalized_mutual_information(self, communities1, communities2):
        c1_labels = [i for i, community in enumerate(communities1) for node in community]
        c2_labels = [i for i, community in enumerate(communities2) for node in community]
        return sk_metrics.normalized_mutual_info_score(c1_labels, c2_labels, average_method='arithmetic')

    def compute_normalized_variation_of_information(self, communities1, communities2):
        c1_labels = [i for i, community in enumerate(communities1) for node in community]
        c2_labels = [i for i, community in enumerate(communities2) for node in community]

        entropy_c1 = self.calculate_entropy(c1_labels)
        entropy_c2 = self.calculate_entropy(c2_labels)
        nmi = sk_metrics.normalized_mutual_info_score(c1_labels, c2_labels)
        vi = (1.0 - nmi) * (entropy_c1 + entropy_c2)
        return vi / np.log2(len(c1_labels))


    def calculate_entropy(self, community):
        # Count the frequency of each type of node
        frequencies = np.bincount(community)
        # Calculate the probabilities
        probabilities = frequencies / len(community)
        # Calculate the entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
        
    def evaluate_communities(self, graph, known_communities, detected_communities, evaluation_function, file_name):
        modularity = nx.algorithms.community.quality.modularity(graph, detected_communities)
        jaccard_index = self.compute_jaccard_index(known_communities, detected_communities)
        normalized_mutual_information = self.compute_normalized_mutual_information(known_communities, detected_communities)
        normalized_variation_of_information = self.compute_normalized_variation_of_information(known_communities, detected_communities)

        return [evaluation_function, file_name,len(detected_communities), modularity, jaccard_index, normalized_mutual_information, normalized_variation_of_information]   
    #-----------------------------------------------------------------------------------------------
    # Plotting functions
    #-----------------------------------------------------------------------------------------------

    def set_kamada_kawai_layout(self, graph):
        self.kamada_kawai_pos = nx.kamada_kawai_layout(graph)

    def plot_kamada_kawai_communities(self, graph, communities, method, file):
        cmap = cm.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, len(communities))]
        plt.figure(figsize=(10, 10))
        plt.title("{method} - {file}" .format(method=method, file=file))
        for community, color in zip(communities, colors):
            nx.draw_networkx_nodes(graph, self.kamada_kawai_pos, nodelist=community, node_color=color)
        nx.draw_networkx_edges(graph, self.kamada_kawai_pos, alpha=0.5)
        plt.show()

    def read_communities_file(self, filename):
        groups = defaultdict(list)
        with open(filename, 'r') as file:
            for line in file:
                node, group = line.strip().split()
                groups[group].append(node)
        return groups



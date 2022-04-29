import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from node2vec import Node2Vec as n2v


def generate_graph_deg_dist(deg_dist, n):
    '''
    This function will generate a networkx graph G based on a degree distribution
    provided by the user.
    
    params:
        deg_dist (Dictionary) : The key will be the degree and the value is the probability
                                of a node having that degree. The probabilities must sum to
                                1
        n (Integer) : The number of nodes you want the graph to yield
                          
    example:
        G = generate_graph_deg_dist(
                deg_dist = {
                    6:0.2,
                    3:0.14,
                    8:0.35,
                    4:0.3,
                    11:0.01
                },
                n = 1000
        )
    '''
    deg = list(deg_dist.keys())
    proba = list(deg_dist.values())
    if sum(proba) == 1.:
        deg_sequence = np.random.choice(
            deg,
            n,
            proba
        )
        
        if sum(deg_sequence) % 2 != 0:
            # to ensure that the degree sequence is always even for the configuration model
            deg_sequence[1] = deg_sequence[1] + 1
            
        return nx.configuration_model(deg_sequence)
    raise ValueError("Probabilities do not equal to 1")
    
def main():
    G = generate_graph_deg_dist(
            deg_dist = {
                6:0.2,
                3:0.14,
                8:0.35,
                4:0.3,
                11:0.01
            },
            n = 150
    )

    print(nx.info(G))

    # visualize degree distribution
    plt.clf()
    plt.hist(list(dict(G.degree()).values()))
    plt.title('Degree Distribution')
    plt.show()

    # visualize the network -- do not run this step if your network is very large with a lot of edges
    nx.draw(G, node_size = 30)

    WINDOW = 1 # Node2Vec fit window
    MIN_COUNT = 1 # Node2Vec min. count
    BATCH_WORDS = 4 # Node2Vec batch words

    g_emb = n2v(
      G,
      dimensions=16
    )

    mdl = g_emb.fit(
        vector_size = 16,
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS
    )

    emb_df = (
        pd.DataFrame(
            [mdl.wv.get_vector(str(n)) for n in G.nodes()],
            index = G.nodes
        )
    )

    X = emb_df.values

    clustering = SpectralClustering(
        n_clusters=5, 
        assign_labels='discretize',
        random_state=0
    ).fit(X)

    print(clustering)


    comm_dct = dict(zip(emb_df.index, clustering.labels_))

    unique_coms = np.unique(list(comm_dct.values()))
    cmap = {
        0 : 'maroon',
        1 : 'teal',
        2 : 'black', 
        3 : 'orange',
        4 : 'green',
    }

    node_cmap = [cmap[v] for _,v in comm_dct.items()]

    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size = 30, alpha = 0.8, node_color=node_cmap)
    plt.show()


if __name__ == '__main__':
    main()
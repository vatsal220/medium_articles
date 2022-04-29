import networkx as nx
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from node2vec import Node2Vec as n2v
from itertools import combinations, groupby
sns.set()

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
            n = 1000
    )
    print(nx.info(G))
    
    # visualize degree dist
    plt.clf()
    plt.hist(list(dict(G.degree()).values()))
    plt.title('Degree Distribution')
    plt.show()
    
    # run n2v
    g_emb = n2v(G, dimensions=16)
    
    WINDOW = 1 # Node2Vec fit window
    MIN_COUNT = 1 # Node2Vec min. count
    BATCH_WORDS = 4 # Node2Vec batch words

    mdl = g_emb.fit(
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS
    )
    
    # convert to df
    emb_df = (
        pd.DataFrame(
            [mdl.wv.get_vector(str(n)) for n in G.nodes()],
            index = G.nodes
        )
    )
    
    # visualize embeddings
    plt.clf()
    fig=plt.figure(figsize=(10,8))
    plt.scatter(
        x = emb_df.iloc[:,0],
        y = emb_df.iloc[:,1],
        s = 0.2
    )
    plt.show()

    # pca 
    pca = PCA(n_components = 2, random_state = 7)
    pca_mdl = pca.fit_transform(emb_df)
    
    emb_df_PCA = (
        pd.DataFrame(
            pca_mdl,
            columns=['x','y'],
            index = emb_df.index
        )
    )
    plt.clf()
    fig = plt.figure(figsize=(6,4))
    plt.scatter(
        x = emb_df_PCA['x'],
        y = emb_df_PCA['y'],
        s = 0.4,
        color = 'maroon',
        alpha = 0.5
    )
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title('PCA Visualization')
    plt.plot()
    
    sns.pairplot(emb_df_PCA)
    
if __name__ == '__main__':
    main()
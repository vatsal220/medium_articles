import networkx as nx
import pandas as pd
import numpy as np
import arxiv

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec as n2v

def search_arxiv(queries, max_results = 100):
    '''
    This function will search arxiv associated to a set of queries and store
    the latest 10000 (max_results) associated to that search.
    
    params:
        queries (List -> Str) : A list of strings containing keywords you want
                                to search on Arxiv
        max_results (Int) : The maximum number of results you want to see associated
                            to your search. Default value is 1000, capped at 300000
                            
    returns:
        This function will return a DataFrame holding the following columns associated
        to the queries the user has passed. 
            `title`, `date`, `article_id`, `url`, `main_topic`, `all_topics`
    
    example:
        research_df = search_arxiv(
            queries = ['automl', 'recommender system', 'nlp', 'data science'],
            max_results = 10000
        )
    '''
    d = []
    searches = []
    # hitting the API
    for query in queries:
        search = arxiv.Search(
          query = query,
          max_results = max_results,
          sort_by = arxiv.SortCriterion.SubmittedDate,
          sort_order = arxiv.SortOrder.Descending
        )
        searches.append(search)
    
    # Converting search result into df
    for search in searches:
        for res in search.results():
            data = {
                'title' : res.title,
                'date' : res.published,
                'article_id' : res.entry_id,
                'url' : res.pdf_url,
                'main_topic' : res.primary_category,
                'all_topics' : res.categories,
                'authors' : res.authors
            }
            d.append(data)
        
    d = pd.DataFrame(d)
    d['year'] = pd.DatetimeIndex(d['date']).year
    
    # change article id from url to integer
    unique_article_ids = d.article_id.unique()
    article_mapping = {art:idx for idx,art in enumerate(unique_article_ids)}
    d['article_id'] = d['article_id'].map(article_mapping)
    return d


def generate_network(df, node_col = 'article_id', edge_col = 'main_topic'):
    '''
    This function will generate a article to article network given an input DataFrame.
    It will do so by creating an edge_dictionary where each key is going to be a node
    referenced by unique values in node_col and the values will be a list of other nodes
    connected to the key through the edge_col.
    
    params:
        df (DataFrame) : The dataset which holds the node and edge columns
        node_col (String) : The column name associated to the nodes of the network
        edge_col (String) : The column name associated to the edges of the network
        
    returns:
        A networkx graph corresponding to the input dataset
        
    example:
        generate_network(
            research_df,
            node_col = 'article_id',
            edge_col = 'main_topic'
        )
    '''
    edge_dct = {}
    for i,g in df.groupby(node_col):
        topics = g[edge_col].unique()
        edge_df = df[(df[node_col] != i) & (df[edge_col].isin(topics))]
        edges = list(edge_df[node_col].unique())
        edge_dct[i] = edges
    
    # create nx network
    g = nx.Graph(edge_dct, create_using = nx.MultiGraph)
    return g

def predict_links(G, df, article_id, N):
    '''
    This function will predict the top N links a node (article_id) should be connected with
    which it is not already connected with in G.
    
    params:
        G (Netowrkx Graph) : The network used to create the embeddings
        df (DataFrame) : The dataframe which has embeddings associated to each node
        article_id (Integer) : The article you're interested 
        N (Integer) : The number of recommended links you want to return
        
    returns:
        This function will return a list of nodes the input node should be connected with.
    '''
    
    # separate target article with all others
    article = df[df.index == article_id]
    
    # other articles are all articles which the current doesn't have an edge connecting
    all_nodes = G.nodes()
    other_nodes = [n for n in all_nodes if n not in list(G.adj[article_id]) + [article_id]]
    other_articles = df[df.index.isin(other_nodes)]
    
    # get similarity of current reader and all other readers
    sim = cosine_similarity(article, other_articles)[0].tolist()
    idx = other_articles.index.tolist()
    
    # create a similarity dictionary for this user w.r.t all other users
    idx_sim = dict(zip(idx, sim))
    idx_sim = sorted(idx_sim.items(), key=lambda x: x[1], reverse=True)
    
    similar_articles = idx_sim[:N]
    articles = [art[0] for art in similar_articles]
    return articles
  
def main():
    '''
    Driver function
    '''
    # constants
    queries = [
        'automl', 'machinelearning', 'data', 'phyiscs','mathematics', 'recommendation system', 'nlp', 'neural networks'
    ]

    WINDOW = 1 # Node2Vec fit window
    MIN_COUNT = 1 # Node2Vec min. count
    BATCH_WORDS = 4 # Node2Vec batch words
    
    # fetch data from arXiv
    research_df = search_arxiv(
        queries = queries,
        max_results = 100
    )
    print(research_df.shape)
    all_tp = research_df.explode('all_topics').copy()
    
    # create network
    tp_nx = generate_network(
        all_tp, 
        node_col = 'article_id', 
        edge_col = 'all_topics'
    )
    print(nx.info(tp_nx))

    # run node2vec
    g_emb = n2v(tp_nx, dimensions=16)

    mdl = g_emb.fit(
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS
    )

    # create embeddings dataframe
    emb_df = (
        pd.DataFrame(
            [mdl.wv.get_vector(str(n)) for n in tp_nx.nodes()],
            index = tp_nx.nodes
        )
    )

    print(emb_df.head())
    
    print("Recommended Links to Article: ", predict_links(G = tp_nx, df = emb_df, article_id = 1, N = 10))

    unique_nodes = list(tp_nx.nodes())
    all_possible_edges = [(x,y) for (x,y) in product(unique_nodes, unique_nodes)]

    # generate edge features for all pairs of nodes
    edge_features = [
        (mdl.wv.get_vector(str(i)) + mdl.wv.get_vector(str(j))) for i,j in all_possible_edges
    ]

    # get current edges in the network
    edges = list(tp_nx.edges())

    # create target list, 1 if the pair exists in the network, 0 otherwise
    is_con = [1 if e in edges else 0 for e in all_possible_edges]
    print(sum(is_con))

    # get training and target data
    X = np.array(edge_features)
    y = is_con

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size = 0.3
    )

    # GBC classifier
    clf = GradientBoostingClassifier()

    # train the model
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_true = y_test

    y_pred = clf.predict(x_test)
    x_pred = clf.predict(x_train)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, x_pred)
    print("Testing Accuracy : ", test_acc)
    print("Training Accuracy : ", train_acc)

    print("MCC Score : ", matthews_corrcoef(y_true, y_pred))

    print("Test Confusion Matrix : ")
    print(confusion_matrix(y_pred,y_test))

    print("Test Classification Report : ")
    print(classification_report(y_test, clf.predict(x_test)))

    pred_ft = [(mdl.wv.get_vector(str('42'))+mdl.wv.get_vector(str('210')))]
    print(clf.predict(pred_ft)[0])

    print(clf.predict_proba(pred_ft))
    
if __name__ == '__main__':
    main()
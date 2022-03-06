import networkx as nx
import pandas as pd
import numpy as np
import arxiv

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
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


def generate_network(df, node_col, edge_col):
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

def clf_eval(clf, x_test, y_test):
    '''
    This function will evaluate a sk-learn multi-class classification model based on its
    x_test and y_test values
    
    params:
        clf (Model) : The model you wish to evaluate the performance of
        x_test (Array) : Result of the train test split
        y_test (Array) : Result of the train test split
    
    returns:
        This function will return the following evaluation metrics:
            - Accuracy Score
            - Matthews Correlation Coefficient
            - Classification Report
            - Confusion Matrix
    
    example:
        clf_eval(
            clf,
            x_test,
            y_test
        )
    '''
    y_pred = clf.predict(x_test)
    y_true = y_test
    
    y_pred = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("Testing Accuracy : ", test_acc)
    
    print("MCC Score : ", matthews_corrcoef(y_true, y_pred))
    
    print("Classification Report : ")
    print(classification_report(y_test, clf.predict(x_test)))
    
    print(confusion_matrix(y_pred,y_test))


def main():

    # constants
    queries = [
        'automl', 'machinelearning', 'data', 'phyiscs','mathematics', 'recommendation system', 'nlp', 'neural networks'
    ]
  
    research_df = search_arxiv(
        queries = queries,
        max_results = 100
    )

    print(research_df.shape)
      
    tp_nx = generate_network(
        research_df, 
        node_col = 'article_id', 
        edge_col = 'main_topic'
    )

    print(nx.info(tp_nx))

    g_emb = n2v(tp_nx, dimensions=16)

    WINDOW = 1 # Node2Vec fit window
    MIN_COUNT = 1 # Node2Vec min. count
    BATCH_WORDS = 4 # Node2Vec batch words

    mdl = g_emb.fit(
        window=WINDOW,
        min_count=MIN_COUNT,
        batch_words=BATCH_WORDS
    )

    emb_df = (
        pd.DataFrame(
            [mdl.wv.get_vector(str(n)) for n in tp_nx.nodes()],
            index = tp_nx.nodes
        )
    )

    emb_df = emb_df.merge(
        research_df[['article_id', 'main_topic']].set_index('article_id'),
        left_index = True,
        right_index = True
    )

    emb_df.head()

    print(emb_df.main_topic.value_counts().head(10))

    ft_cols = emb_df.drop(columns = ['main_topic']).columns.tolist()
    target_col = 'main_topic'

    # train test split
    x = emb_df[ft_cols].values
    y = emb_df[target_col].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y,
        test_size = 0.3
    )

    # GBC classifier
    clf = GradientBoostingClassifier()

    # train the model
    clf.fit(x_train, y_train)
        
    # evaluate model
    clf_eval(
        clf,
        x_test,
        y_test
    )

    # predict node classes using classification model
    pred_ft = [mdl.wv.get_vector(str('21'))]
    print(clf.predict(pred_ft)[0])
    

if __name__ == '__main__':
    main()
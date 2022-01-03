import pandas as pd
import numpy as np
import networkx as nx
import random
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
    
from pylab import rcParams
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from collections import defaultdict
sns.set()

# constants
names_df_path = './data/ontario_names_list_1917-2019.csv'
stop = stopwords.words('english')
additional_sw = [
    'well', 'would', 'never', 'latitude', 'longitude', 'wonderland', 'wonder', 'adventure', 'adventures', 'chapter',
    'please', 'maam', 'drink', 'think', 'sink', 'come', 'foot', 'right', 'thats', 'too', 'itll', 'tell', 'table', 'long',
    'tale', 'test', 'said', 'held', 'crab', 'next', 'sure', 'digging', 'i', 'i\'ve', 'oh', 'bills',
    'number', 'ill', 'im', 'ive', 'whats', 'read', 'little', 'id',    'look', 'dont', 'luckily', 'get', 'lizard', 'the'
]

# import data
# to download books data
nltk.download('gutenberg')
book = nltk.corpus.gutenberg.raw('carroll-alice.txt')

def remove_punctuations(txt, punct = string.punctuation):
    '''
    This function will remove punctuations from the input text
    '''
    return ''.join([c for c in txt if c not in punct])
  
def remove_stopwords(txt, sw = list(stopwords.words('english')) + additional_sw):
    '''
    This function will remove the stopwords from the input txt
    '''
    return ' '.join([w for w in txt.split() if w.lower() not in sw])

def clean_text(txt):
    '''
    This function will clean the text being passed by removing specific line feed characters
    like '\n', '\r', and '\'
    '''
    
    txt = txt.replace('\n', ' ').replace('\r', ' ').replace('\'', '')
    txt = remove_punctuations(txt)
    txt = remove_stopwords(txt)
    return txt.lower()
  
book = clean_text(book) 

# find character names 
# male names source : https://data.ontario.ca/dataset/ontario-top-baby-names-male
# female names source : https://data.ontario.ca/dataset/ontario-top-baby-names-female

names_df = pd.read_csv(
    names_df_path
)

names_list = list(names_df['name'].values)

def find_names(text, given_names, names):
    '''
    Given a body of text, this function will identify the names in that body of text
    by cross referencing list of names in the body of text
    
    params:
        text (String) : A body of text you want to parse through to identify names
        given_names (List / Dict) : The list / dictionary of known names in the body of 
                                    text, if the input is a dictionary, the keys are the
                                    main names and the values are a list of aliases
        names (List) : A corpus of names recorded throughout the world
        
    returns:
        This function will return an updated given_names which holds the user input as 
        well as other names which the algorithm has found
    '''
    text = text.lower().split(' ')
    given_names_copy = given_names
    kn = []
    if type(given_names) == dict:
        for k,v in given_names_copy.items():
            if k not in v:
                given_names_copy[k].append(k)
        for k,v in given_names_copy.items():
            for name in v:
                kn.append(name)
    elif type(given_names) == list:
        for name in names:
            if name in text and name not in given_names:
                given_names.append(name)
        return given_names
        
    for name in names:
        if name in text and name not in kn:
            given_names_copy[name] = []
    return given_names_copy

known_names = {
    'alice' : [],
    'white rabbit' : ['rabbit'],
    'mouse' : [],
    'dodo' : [],
    'lory' : [],
    'eaglet' : [],
    'duck' : [],
    'pat' : [],
    'bill the lizzard' : ['bill', 'lizard'],
    'puppy' : [],
    'caterpillar' : [],
    'duchess' : [],
    'cheshire cat' : ['cat', 'cheshire'],
    'hatter' : [],
    'march hare' : ['hare', 'march'],
    'dormouse' : [],
    'queen of hearts' : ['queen'],
    'king of hearts' : ['king'],
    'knave of hearts' : ['knave'],
    'gryphon' : [],
    'mock turtle' : ['turtle'],
    'two' : [],
    'five' : [],
    'seven' : []
}

kn = [
    'alice', 'rabbit', 'queen', 'king', 'cat', 'Duchess', 'caterpillar', 'hatter', 'hare', 'dormouse', 'gryphon',
    'turtle', 'sister', 'knave', 'mouse', 'dodo', 'cook', 'duck', 'pigeon', 'two', 'five', 'seven', 'bill', 'frog'
]
found_names_lst = find_names(book, given_names = kn, names = names_list)

found_names_dct = find_names(book, given_names = known_names, names = names_list)
print(len(found_names_lst), len(found_names_dct.keys()))
print(found_names_lst)

# identify character interactions
N = 15
text = book
found_names = found_names_dct

def lower_known_names_dct(known_names):
    '''
    The purpose of this function is to add the key into the list of values
    as a lowered key
    '''
    known_names_full = known_names.copy()
    for k,v in known_names_full.items():
        if k.lower not in v:
            v.append(k.lower())
    return known_names_full

found_names_full = lower_known_names_dct(known_names)

def generate_idx_dict(text, found_names_full):
    '''
    create an idx_dct where the keys are the idx of the names in
    the text and the values are the names
    '''
    res = dict()
    for i,word in enumerate(text.lower().split(' ')):
        for k,v in found_names_full.items():
            if word not in v:
                continue
            res[i] = k
    return res
idx_dct = generate_idx_dict(text, found_names_full)

def find_interactions(idx_dct, N):
    res = defaultdict(int)
    names = list(idx_dct.keys())
    
    for i,na in enumerate(names):
        # given an index, get the sublist of all indicies greater than the current index
        if i < len(names) - 1:
            kl = names[i+1:]
        else:
            kl = []
        
        # for each idx greater than the current, check if its found in the range of N
        for k in kl:
            if k-na < N:
                # get names found in current position (na) and index greater than current but in rnage N (k)
                n1 = idx_dct[na]
                n2 = idx_dct[k]
                
                key = tuple(sorted([n1,n2]))
                res[key]+=1
    return res
interactions_dct = find_interactions(idx_dct, N)
interactions_lst = list([(*k,v) for (k,v) in interactions_dct.items()])

# create network
edges_df = pd.DataFrame(interactions_lst, columns = ['source', 'target', 'weight'])
G = nx.from_pandas_edgelist(edges_df, edge_attr = True)
print(nx.info(G))

# visualize network
rcParams['figure.figsize'] = 14, 10
pos = nx.circular_layout(G, scale = 20)
labels = nx.get_edge_attributes(G, 'weight')
# pos = nx.spring_layout(G, scale=20, k=3/np.sqrt(G.order()))
d = dict(G.degree)
nx.draw(
    G, 
    pos,
    node_color='lightblue', 
    alpha = 0.75,
    with_labels=True, 
    nodelist=d, 
    node_size=[d[k]*200 for k in d],
    edgelist = labels
)
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.show()

pr = nx.pagerank(G)
pr = {k: v for k, v in sorted(pr.items(), reverse = True, key=lambda item: item[1])}
pr_df = pd.DataFrame([pr]).T.reset_index().rename(columns = {0 : 'pr', 'index' : 'name'})

# visualize page rank to see character importance
plt.barh(y = pr_df['name'].head(10), width = pr_df['pr'].head(10))
plt.title("Page Rank of Alice in Wonderland Network")
plt.ylabel("Characters")
plt.xlabel("Page Rank Score")
plt.show()

# chung lu model
# source : https://github.com/ftudisco/scalefreechunglu/blob/master/python/fastchunglu.py
# The two functions below generate an instance of the Chung-Lu random graph model with expected degree sequence w

def make_sparse_adj_matrix(w):
    '''
    This function creates a sparse adjacency matrix of the input vector w
    
    params:
        w (List) :  A vector of nonnegative real numbers which define the model
    
    returns:
        It will return a scipy.sparse adjacency matrix of the graph
    '''
    # Outputs the scipy.sparse adjacency matrix of the graph
    n = np.size(w)
    s = np.sum(w)
    m = ( np.dot(w,w)/s )**2 + s
    m = int(m/2)
    wsum = np.cumsum(w)
    wsum = np.insert(wsum,0,0)
    wsum = wsum / wsum[-1]
    I = np.digitize(np.random.rand(m,1),wsum)
    J = np.digitize(np.random.rand(m,1),wsum)
    row_ind = np.append(I.reshape(m,)-1,J.reshape(m,)-1)
    col_ind = np.append(J.reshape(m,1)-1,I.reshape(m,)-1)
    ones = [1 for i in range(2*m)]    
    A = csr_matrix((ones, (row_ind,col_ind)), shape=(n,n))
    A.data.fill(1)
    return A


def make_nx_graph(w):
    '''
    This function will create a networkx graph from the input vector w
    
    params:
        w (List) :  A vector of nonnegative real numbers which define the model
    
    returns:
        It will return a nx.Graph of the graph
    '''
    n = np.size(w)
    s = np.sum(w)
    m = ( np.dot(w,w)/s )**2 + s
    m = int(m/2)
    wsum = np.cumsum(w)
    wsum = np.insert(wsum,0,0)
    wsum = wsum / wsum[-1]
    I = np.digitize(np.random.rand(m,1),wsum)
    J = np.digitize(np.random.rand(m,1),wsum)
    G = nx.Graph()
    G.add_nodes_from(range(1,n+1))
    G.add_edges_from(tuple(zip(I.reshape(m,),J.reshape(m,))))
    return G
  
degrees = [x[1] for x in list(G.degree(weight='weight'))]
avg_deg = sum(degrees) / float(len(G))
n = len(G.nodes())
gamma = 10       # power law distribution exponent
m = max(degrees)  # max degree
d = avg_deg   # avg degree
p = 1/(gamma-1)
c = (1-p)*d*(n**p)
i0 = (c/m)**(1/p) - 1
w = [c/((i+i0)**p) for i in range(1, n+1)]

A = make_sparse_adj_matrix(w)
G_cl = make_nx_graph(w)
print(nx.info(G_cl))
print(nx.info(nx.expected_degree_graph(w)))

def main():
    pass
    
if __name__ == '__main__':
    main()
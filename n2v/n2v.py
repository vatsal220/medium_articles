import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, spdiags, hstack, lil_matrix
from scipy.sparse.linalg import inv, norm
from gensim.models import Word2Vec


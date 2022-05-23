import pandas as pd
import numpy as np
from numpy.linalg import svd
from random import randint

def generate_data(n_songs = 1000, n_genres = 5, n_artists = 500, n_users = 3000, n_listens = 15, dataset_size = 100000):
    '''
    This function will generate a dataset with features associated to
    song data set. The dataset will have the following columns : 
        - song_id (String) : Unique identified for the song
        - user_id (String) : Unique identifier for the user
        - song_genre (Integer) : An integer representing a genre for the song, 
                                 value is between 1 and 15, indicating that 
                                 there are 15 unique genres. Each song can only
                                 have 1 genre
        - artist_id (String) : Unique identifier for the author of the song
        - n_listen (Integer) : The number of times this user has heard the song
        - publish_year (Integer) : The year of song publishing
        
    params:
        n_songs (Integer) : The number of songs you want the dataset to have
        n_genres (Integer) : Number of genres to be chosen from
        n_artists (Integer) : Number of authors to be generated
        n_users (Integer) : Number of readers for the dataset
        n_listens (Integer) : Range of number of times a song has been heard
        dataset_size (Integer) : The number of rows to be generated 
        
    example:
        data = generate_data()
    '''
    
    d = pd.DataFrame(
        {
            'song_id' : [randint(1, n_songs) for _ in range(dataset_size)],
            'artist_id' : [randint(1, n_artists) for _ in range(dataset_size)],
            'song_genre' : [randint(1, n_genres) for _ in range(dataset_size)],
            'user_id' : [randint(1, n_users) for _ in range(dataset_size)],
            'n_listen' : [randint(0, n_listens) for _ in range(dataset_size)],
            'publish_year' : [randint(2000, 2021) for _ in range(dataset_size)]
        }
    ).drop_duplicates()
    return d

def cosine_similarity(a, b):
    '''
    This function will calculate the cosine similarity between two vectors
    '''
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_similarities(mat, id_):
    '''
    This function will use the cosine similarity function to generate a similarity 
    dictionary assocaited to an id the user passes in. The similarity dictionary will
    have the ids as the keys and the similarity in comparison to the user input id as
    the values.
    
    params:
        mat (List -> List) : A 2-D array assocaited to either the user / item matrix
                             after SVD
        id_ (Integer) : The id of the user / item you want to find similarities for.
                        The id must be in the range of the input matrix shape.
    
    returns:
        This function will return the similarity dictionary ordered by the values in 
        descending order.
        
    example:
        mat = np.asarray([
            [2,3,4],
            [6,5,3],
            [5,3,2]
        ])
        id_ = 2
        get_similarities(mat, id_)
    '''
    # create similarity hashmap, keys are ids and values are similarities
    sim_dct = {} 
    for col in range(0, mat.shape[1]):
        sim = cosine_similarity(mat[:,id_], mat[:,col])
        sim_dct[col] = sim
    
    # sort dictionary based on similarities 
    sim_dct = {k: v for k, v in sorted(sim_dct.items(), key=lambda item: item[1], reverse = True)}
    return sim_dct

def recommend(mat, id_, n_recs):
    '''
    This function will get the top n recommendations assocaited to an id.
    
    params:
        mat (List -> List) : A 2-D array assocaited to either the user / item matrix
                             after SVD
        id_ (Integer) : The id of the user / item you want to find similarities for.
                        The id must be in the range of the input matrix shape.
        n_recs (Integer) : The number of recommendations you want.
        
    returns:
        This function will return a list of ids most similar to the input id you passed.
    '''
    sim_dct = get_similarities(mat, id_)
    similar_ids = list(sim_dct.keys())[1:n_recs+1]
    return similar_ids


def main():
    # generate data
    d = generate_data(dataset_size = 100000).drop_duplicates()
    d.to_csv('data.csv', index = False)

    # convert to user-item matrix
    item_col = 'song_id'
    user_col = 'user_id'
    freq_col = 'n_listen'

    mat = d.pivot_table(
        index = user_col, 
        columns = item_col, 
        values = freq_col, 
        fill_value=0, 
        aggfunc = 'mean'
    )
    print(mat.shape)
    
    # calculate SVD
    u, sigma, v = svd(mat.values)
    
    # generate recommendations
    id_ = 8
    n_recs = 10
    print(recommend(u, id_, n_recs))
    
if __name__ == '__main__':
    main()
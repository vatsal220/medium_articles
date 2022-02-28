import networkx as nx
import numpy as np
import nltk
import jaro

try:
    from nltk.corpus import stopwords
except:
    import nltk
    nltk.download('stopwords')
finally:
    from nltk.corpus import stopwords

try:
    book = nltk.corpus.gutenberg.raw('shakespeare-caesar.txt')
except:
    nltk.download('gutenberg')
finally:
    book = nltk.corpus.gutenberg.raw('shakespeare-caesar.txt')
    
def clean_text(text, sw, punct):
    '''
    This function will clean the input text by lowering, removing certain punctuations, stopwords and 
    new line tags.
    
    params:
        text (String) : The body of text you want to clean
        sw (List) : The list of stopwords you wish to removed from the input text
        punct (List) : The slist of punctuations you wish to remove from the input text
        
    returns:
        This function will return the input text after it's cleaned (the output will be a string)
    '''
    text = text.lower()
    article = text.split(' ')
    # clean stopwords
    article = [x.lstrip().rstrip() for x in article if x not in sw]
    article = [x for x in article if x]
    article = ' '.join(article)
    
    # clean punctuations
    for pun in punct:
        article = article.replace(pun, '')
    
    article = article.replace("[^a-zA-Z]", " ").replace('\r\n', ' ').replace('\n', ' ')
    return article

def create_similarity_matrix(sentences):
    '''
    The purpose of this function will be to create an N x N similarity matrix.
    N represents the number of sentences and the similarity of a pair of sentences
    will be determined through the Jaro-Winkler Score.
    
    params:
        sentences (List -> String) : This is a list of strings you want to create
                                     the similarity matrix with.
     
    returns:
        This function will return a square numpy matrix
    '''
    
    # identify sentence similarity matrix with Jaro Winkler score
    sentence_length = len(sentences)
    sim_mat = np.zeros((sentence_length, sentence_length))

    for i in range(sentence_length):
        for j in range(sentence_length):
            if i != j:
                similarity = jaro.jaro_winkler_metric(sentences[i], sentences[j])
                sim_mat[i][j] = similarity
    return sim_mat

def generate_summary(ranked_sentences, N):
    '''
    This function will generate the summary given a list of ranked sentences and the
    number of sentences the user wants in their summary.
    
    params:
        ranked_sentences (List -> Tuples) : The list of ranked sentences where each
                                            element is a tuple, the first value in the
                                            tuple is the sentence, the second value is
                                            the rank
        N (Integer) : The number of sentences the user wants in the summary
        
    returns:
        This function will return a string associated to the summarized ranked_sentences
        of a book
    '''
    summary = '. '.join([sent[0] for sent in ranked_sentences[0:N]])
    return summary

def main():
    # constants
    sw = list(set(stopwords.words('english')))
    punct = [
        '!','#','$','%','&','(',')','*',
        '+',',','-','/',':',';','<','=','>','@',
        '[','\\',']','^','_','`','{','|','}','~'
    ]
    
    # clean data
    cleaned_book = clean_text(book, sw, punct)

    # get sentences
    sentences = [x for x in cleaned_book.split('. ') if x not in ['', ' ', '..', '.', '...']]
    print(len(sentences))
    
    # generate similarity matrix
    sim_mat = create_similarity_matrix(sentences)

    # create network
    G = nx.from_numpy_matrix(sim_mat)

    # calculate page rank scores
    pr_sentence_similarity = nx.pagerank(G)

    ranked_sentences = [
        (sentences[sent], rank) for sent,rank in sorted(pr_sentence_similarity.items(), key=lambda item: item[1], reverse = True)
    ]

    print(ranked_sentences[0])
  
    # create summary
    N = 25
    summary = generate_summary(ranked_sentences, N)
    print(summary)
    return summary

if __name__ == '__main__':
    main()
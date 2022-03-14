import string
import Levenshtein as lev
import wikipedia

from functools import lru_cache
from nltk.corpus import stopwords

def lev_dist(a, b):
    '''
    This function will calculate the levenshtein distance between two input
    strings a and b
    
    params:
        a (String) : The first string you want to compare
        b (String) : The second string you want to compare
        
    returns:
        This function will return the distnace between string a and b.
        
    example:
        a = 'stamp'
        b = 'stomp'
        lev_dist(a,b)
        >> 1.0
    '''
    
    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)

def fetch_wiki_data(tags):
    '''
    The purpose of this function is to get the wikipedia data associated to a certain user
    input tag.
    
    params:
        tag (String) : The item you want to seach wikipedia for
        
    returns:
        This function will return the contents associated to the user specified tag
    
    example:
        tag = 'Levenshtein distance'
        fetch_wiki_data(tag)
        >> In information theory, linguistics, and computer science, the Levenshtein distance 
           is a string metric...
    '''
    content = {}
    for tag in tags:
        # get wikipedia data for the tag
        wiki_tag = wikipedia.search(tag)

        # get page info
        page = wikipedia.page(wiki_tag[0])

        # get page content
        content[tag] = page.content
    return content

def remove_punctuations(txt, punct = string.punctuation):
    '''
    This function will remove punctuations from the input text
    '''
    return ''.join([c for c in txt if c not in punct])
  
def remove_stopwords(txt, sw = list(stopwords.words('english'))):
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
    
def similarity(user_article, tag_content):
    '''
    This function will identify the similarities between the user_article and all the
    content within tag_content
    
    params:
        user_article (String) : The text submitted by the user
        tag_content (Dictionary) : Key is the tag and the value is the content you want 
                                   to compare with the user_article
    
    returns:
        This function will return a dictionary holding the Levenshtein assocaited to the 
        user_article with each tag_content
    '''
    
    distances = {}
    for tag,content in tag_content.items():
        dist = lev.distance(user_article, content)
        distances[tag] = dist
    return distances
  
def is_plagiarism(user_article, tag_content, distances, th = 0.4):
    '''
    This function will identify if the user_article is considered plagiarized for each
    of the tag_content based on the distances observed.
    
    params:
        user_article (String) : The text submitted by the user
        tag_content (Dictionary) : Key is the tag and the value is the content you want 
                                   to compare with the user_article
        distances (Dictionary) : Key is the tag and the value is the Levenshtein distance 
        th (Float) : The plagiarism threshold
    
    returns:
        A dictionary associated to the plagiarism percentage for each tag
    '''
    ua_len = len(user_article)
    distances = {tag:[d, max(ua_len, len(tag_content[tag]))] for tag,d in distances.items()}
    
    for tag, d in distances.items():
        if d[0] <= d[1] * th:
            distances[tag] = 'Plagiarized'
        else:
            distances[tag] = 'Not Plagiarized'
    return distances
  
def main():
    # Implemented Lev Distance
    a = 'stamp'
    b = 'stomp'
    print(a, b)
    print(lev_dist(a,b))

    # Plagiarism Pipeline
    user_article = '''
    Identifying similarity between text is a common problem in NLP and is used by many companies world wide. The most common application of text similarity comes from the form of identifying plagiarized text. Educational facilities ranging from elementary school, high school, college and universities all around the world use services like Turnitin to ensure the work submitted by students is original and their own. Other applications of text similarity is commonly used by companies which have a similar structure to Stack Overflow or Stack Exchange. They want to be able to identify and flag duplicated questions so the user posting the question can be referenced to the original post with the solution. This increases the number of unique questions being asked on their platform. 
    Text similarity can be broken down into two components, semantic similarity and lexical similarity. Given a pair of text, the semantic similarity of the pair refers to how close the documents are in meaning. Whereas, lexical similarity is a measure of overlap in vocabulary. If both documents in the pairs have the same vocabularies, then they would have a lexical similarity of 1 and vice versa of 0 if there was no overlap in vocabularies [2].
    Achieving true semantic similarity is a very difficult and unsolved task in both NLP and Mathematics. It's a heavily researched area and a lot of the solutions proposed does involve a certain degree of lexical similarity in them. For the focuses of this article, I will not dive much deeper into semantic similarity, but focus a lot more on lexical similarity.
    Levenshtein Distance
    There are many ways to identify the lexical similarities between a pair of text, the one which we'll be covering today in this article is Levenshtein distance. An algorithm invented in 1965 by Vladimir Levenshtein, a Soviet mathematician [1].
    Intuition
    Levenshtein distance is very impactful because it does not require two strings to be of equal length for them to be compared. Intuitively speaking, Levenshtein distance is quite easy to understand.
    Informally, the Levenshtein distance between two words is the minimum number of single-character edits (insertions, deletions or substitutions) required to change one word into the other. [1]
    - https://en.wikipedia.org/wiki/Levenshtein_distance
    Essentially implying that the output distance between the two is the cumulative sum of the single-character edits. The larger the output distance is implies that more changes were necessary to make the two words equal each other, and the lower the output distance is implies that fewer changes were necessary. For example, given a pair of words dream and dream the resulting Levenshtein distance would be 0 because the two words are the same. However, if the words were dream and steam the Levenshtein distance would be 2 as you would need to make 2 edits to change dr to st .
    Thus a large value for Levenshtein distance implies that the two documents were not similar, and a small value for the distance implies that the two documents were similar.
    '''

    tags = ['Levenshtein distance']
    tag_content = fetch_wiki_data(tags)

    user_article = clean_text(user_article) 
    for tag, content in tag_content.items():
        tag_content[tag] = clean_text(content)
        
    distances = similarity(user_article, tag_content)
    
    print(is_plagiarism(user_article, tag_content, distances))

if __name__ == '__main__':
    main()
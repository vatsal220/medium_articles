import os
import bs4
from urllib import request

# constants
res_path = "./results/"
url = "https://www.gutenberg.org/files/1342/1342-0.txt"


def mkdir(path="./results/"):
    """
    This function will create a directory given a path if one does
    not already exist in that specified path.
    
    params:
        path (String) : The location you want to create the directory
        
    returns:
        This function will create a folder called results if one does
        not already exist
        
    example:
        mkdir(
            path = './results/'
        )
    """
    if not os.path.exists(path):
        os.makedirs(path)


def getHTML(url):
    """
    This function will fetch the HTML associated with a list of urls.
    
    params:
        url (String) : The url you want to fetch
        
    returns:
        A BeatifulSoup object associated with the url you want requested.
    """
    res = request.urlopen(url)
    soup = bs4.BeautifulSoup(res.read().decode("utf8"))
    return soup


def save_book(soup, save_path=res_path + "book.txt"):
    """
    The purpose of this function will be to save the location associated with
    a book scraped from Project Gutenberg.
    
    params:
        soup (bs4) : The soup associated with a url
        save_path (String) : The location where you want to save the results
                             of the scrape
                             
    returns:
        This function will not return anything, it will create & write to a 
        text file associated with the text contents from the soup
    """
    with open(res_path + "/book.txt", "w") as f:
        f.write(soup.text)


def main(url):
    """
    Driver function
    """
    mkdir()
    soup = getHTML(url)
    save_book(soup, save_path=res_path + "book.txt")


if __name__ == "__main__":
    main(url)

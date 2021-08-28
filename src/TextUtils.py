from textacy.preprocessing.normalize import hyphenated_words, quotation_marks, unicode, \
    whitespace
from textacy.preprocessing.replace import urls
from textacy.preprocessing.remove import punctuation

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from urllib.parse import urlparse
import re

def clean_text(t):
    t = hyphenated_words(t)
    t = unicode(t)
    t = t.replace("\n", "")
    t = whitespace(t)
    t = urls(t)
    t = punctuation(t)
    t = ' '.join([word for word in t.split() if word not in ENGLISH_STOP_WORDS])
    t = t.lower()
    return t

# Function to split URLs from crawl results and generate hierarchy
def url_splitter(url):
    # Pass url to parser
    u = urlparse(url)
    # take path (without netloc) and remove leading /
    path = u.path[1:]
    path_list = path.split("/")
    path_list.remove("editions-pricing")
    hierarchy = '/'.join(path_list)[:-1]
    return hierarchy


# Normalize url for seed pages
def clean_seed_url(url):
    # Derive url path after base and take first level of product path
    cleaned_url = (url_splitter(url) + "/").split("/")[0]
    return cleaned_url


# Normalize urls for help articles
def clean_article_url(url):
    cleaned_url = re.sub(r'\&(.*)', '', (re.sub(r'^.*?\?', '', url)))
    return cleaned_url
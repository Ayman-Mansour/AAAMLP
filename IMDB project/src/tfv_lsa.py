# src/tfv_lsa.py

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from nltk.tokenize import word_tokenize
import re
import string

def clean_text(s):
    """
    This function clean the text a bit 
    :param s: string
    :return: cleaned string
    """
    # split by all whitespaces
    s = s.split()
    # join tokens by single space
    # why we do this?
    # this will remove all kinds of weird space
    # "hi.   how are you" becomes
    # "hi. how are you"
    s = " ".join(s)
    # remove all punctuations using regex and string module
    s = re.sub(f'[{re.escape(string.punctuation)}]', '', s)
    # you can add more cleaning here if you want
    # and then return the cleaned string
    return s

corpus = pd.read_csv("../input/imdb.csv", nrows=10000)
corpus.loc[:, "review"] = corpus.review.apply(clean_text)
corpus = corpus.review.values
        
tfv = TfidfVectorizer(
    tokenizer=word_tokenize,
    token_pattern=None
)

tfv.fit(corpus)

corpus_transformed = tfv.transform(corpus)

svd = decomposition.TruncatedSVD(n_components=10)
    
corpus_svd = svd.fit(corpus_transformed)

sample_index = 0
feature_scores = dict(
    zip(
        tfv.get_feature_names(),
        corpus_svd.components_[sample_index]
    )
)

N = 5
print(sorted(feature_scores, key=feature_scores.get, reverse=True)[:N])


N= 5

for sample_index in range(5):
    feature_scores = dict(
        zip(
        tfv.get_feature_names(),
        corpus_svd.components_[sample_index]
        )
    )
    
    print(
        sorted(
            feature_scores, 
            key=feature_scores.get,
            reverse=True)[:N]
    )
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

corpus = ["hello, how are you?",
          "im getting bored at home. And you? What do you think?",
          "did you know about counts",
          "let's see if this works!",
          "YES!!!!"
         ]

ctv = CountVectorizer()

ctv.fit(corpus)

corpus_transformed = ctv.transform(corpus)

print(corpus_transformed)

print(ctv.vocabulary_)

#####################################################################

ctv_t = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

ctv_t.fit(corpus)

corpus_transformed_t = ctv_t.transform(corpus)

print(corpus_transformed_t)

print(ctv_t.vocabulary_)

#####################################################################


ctv_tidf = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)

ctv_tidf.fit(corpus)

corpus_transformed_tidf = ctv_tidf.transform(corpus)

print(corpus_transformed_tidf)

print(ctv_tidf.vocabulary_)

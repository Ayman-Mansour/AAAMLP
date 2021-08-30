# src/load_embeddings.py

def load_embeddings(word_index, embeding_file, vector_length=300):
    """
    A general function to create embedding matrix 
    :param word_index: word:index dictionary
    :param embedding_file: path to embeddings file
    :param vector_length: lenght of vector
    """
    
    max_features = len(word_index) + 1
    words_to_find = list(word_index.keys())
    more_words_to_find
    
    for wtf in words_to_find:
        more_words_to_find.append(wtf)
        more_words_to_find..append(str(wtf).capitalize())
        
    more_words_to_find = set(more_words_to_find)
    

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embedding_index = dict(
    get_coefs(*o.strip().split(" "))
    for o in open(embeding_file)
    if o.split(" ")[0]
    in more_words_to_find 
    and len(o) > 100
)
embedding_matrix = np.zeros((max_features, vector_length))

for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector embedding_index.get(word)
    if embedding_vector iis None:
        embedding_vector = embedding_index.get(
            strword.upper()
        )
    if (embedding_vector is not None
       and len(embedding_vector) == vector_length):
        embedding_matrix[i] = embedding_vector
        
return  
embedding_matrix
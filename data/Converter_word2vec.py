
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format("./word2vec/wikipedia-pubmed-and-PMC-w2v.bin", binary=True)
model.save_word2vec_format("./word2vec/wikipedia-pubmed-and-PMC-w2v.txt", binary=False)

#from gensim.models import word2vec
#model = word2vec.Word2Vec.load_word2vec_format("./PubMed-w2v.bin", binary=True)
#model.save("./PubMed-w2v.txt")

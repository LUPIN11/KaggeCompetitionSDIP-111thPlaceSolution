from gensim import utils

class Corpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        # corpus_path = datapath()
        for line in open('text8.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

import gensim.models

sentences = Corpus()
model = gensim.models.Word2Vec(sentences=sentences,
                               min_count=5,
                               vector_size=100,
                               workers=3,
                               )


w1 = 'women'
w2 = 'men'

print(f'cs of {w1} and {w2}:', model.wv.similarity(w1, w2))
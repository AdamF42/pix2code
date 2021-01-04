from typing import List

import numpy as np
from gensim.models import Word2Vec

from utils.vocabulary import Vocabulary


class VocabularyW2V(Vocabulary):

    def __init__(self, w2v_model: Word2Vec):
        self.w2v_model = w2v_model

    def word_to_index(self, word):
        return self.w2v_model.wv.vocab[word].index

    def word_to_encode(self, word):
        return self.w2v_model.wv[word]

    def index_to_word(self, index):
        return self.w2v_model.wv.index2word[index]

    def get_tokens(self):
        return self.w2v_model.wv.vocab.keys()

    def w2v_encode(self, sentence) -> List[np.ndarray]:
        if isinstance(sentence, list):
            return [self.word_to_encode(word) for word in sentence]
        else:
            return [self.word_to_encode(word) for word in sentence.split(" ")]

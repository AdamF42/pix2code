from gensim.models import Word2Vec

from utils.vocabulary import Vocabulary


class VocabularyW2V(Vocabulary):

    def __init__(self, w2v_model: Word2Vec):
        self.w2v_model = w2v_model

    def word_to_index(self, word):
        return self.w2v_model.wv.vocab[word].index

    def index_to_word(self, index):
        return self.w2v_model.wv.index2word[index]

    def get_tokens(self):
        return self.w2v_model.wv.vocab.keys()
from utils.vocabulary import Vocabulary


class VocabularyOneHot(Vocabulary):

    def __init__(self, word_to_encode: dict):
        self.words = list(word_to_encode.keys())
        self.w2encode = word_to_encode
        self.w2i = {val: key for key, val in enumerate(word_to_encode)}
        self.i2w = {val: key for key, val in self.w2i.items()}

    def word_to_index(self, word):
        return self.w2i[word]

    def word_to_encoding(self, word):
        return self.w2encode[word]

    def index_to_word(self, index):
        return  self.i2w[index]

    def get_tokens(self):
        return self.words

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import sys

import numpy as np

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PLACEHOLDER = " "
SEPARATOR = '->'


class Vocabulary:
    def __init__(self):
        self.binary_vocabulary = {}
        self.vocabulary = {}  # {'<START>': 0, '<END>': 1, ' ': 2, 'header': 3, . . . }
        self.token_lookup = {}  # {0: '<START>', 1: '<END>', 2: ' ', 3: 'header', . . . }
        self.inv_token_lookup = {}  # {'<START>': 0, '<END>': 1, ' ': 2, 'header': 3, '{': 4 . . .
        self.size = 0

        self.append(START_TOKEN)
        self.append(END_TOKEN)
        self.append(PLACEHOLDER)

    def append(self, token):
        if token not in self.vocabulary:
            self.vocabulary[token] = self.size
            self.token_lookup[self.size] = token
            self.size += 1

    def create_binary_representation(self):
        if sys.version_info >= (3,):
            items = self.vocabulary.items()
        else:
            items = self.vocabulary.iteritems()
        for key, value in items:
            binary = np.zeros(self.size)
            binary[value] = 1
            self.binary_vocabulary[key] = binary

    def get_serialized_binary_representation(self):
        # TODO: create two Voc classes: one for w2c and one for one_hot
        # if len(self.binary_vocabulary) == 0:
        #     self.create_binary_representation()

        string = ""
        if sys.version_info >= (3,):
            items = self.binary_vocabulary.items()
        else:
            items = self.binary_vocabulary.iteritems()
        for key, value in items:
            print("Max line size: "+ str(self.size * value.size))
            max_line_width=self.size * value.size
            array_as_string = np.array2string(value, separator=',', max_line_width=max_line_width)
            string += "{}{}{}\n".format(key, SEPARATOR, array_as_string[1:len(array_as_string) - 1])
        return string  # \n->[codifica]\n

    def save(self, path):
        output_file_name = "{}/words.vocab".format(path)
        output_file = open(output_file_name, 'w')
        output_file.write(self.get_serialized_binary_representation())
        output_file.close()

    def retrieve(self, path):
        input_file = open("{}/words.vocab".format(path), 'r')
        buffer = ""
        index = 0
        for line in input_file:
            try:
                separator_position = len(buffer) + line.index(SEPARATOR)
                buffer += line
                key = buffer[:separator_position]
                value = buffer[separator_position + len(SEPARATOR):]
                value = np.fromstring(value, sep=',')

                self.binary_vocabulary[key] = value
                self.vocabulary[key] = value  # np.where(value == 1)[0][0]
                self.token_lookup[index] = key
                self.inv_token_lookup = {v: k for k, v in self.token_lookup.items()}

                buffer = ""
                index += 1
            except ValueError:
                buffer += line
        input_file.close()
        self.size = len(self.vocabulary)

#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import sys

from classes.Sampler import *
from classes.model.pix2code import *

import numpy as np

from classes.dataset.Dataset import Dataset

argv = sys.argv[1:]

if len(argv) < 5:
    print("Error: not enough argument supplied:")
    print("generate.py <trained weights path> <trained model name> <input image> <output path> <encoding_type> <search method (default: greedy)>")
    exit(0)
else:
    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    input_path = argv[2]
    output_path = argv[3]
    encoding_type = argv[4]
    search_method = "greedy" if len(argv) < 6 else argv[5]

meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
input_shape = meta_dataset[0]
output_size = meta_dataset[1]

model = pix2code(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

dataset = Dataset()
if encoding_type == "one_hot":
    dataset.load_with_one_hot_encoding(input_path, generate_binary_sequences=True)
elif encoding_type == "w2v":
    dataset.load_with_word2vec(input_path, generate_binary_sequences=True)
else:
    raise Exception("Missing parameter")

# dataset = Dataset()
# dataset.load(input_path))

''' Metodo di valutazione cinese
model.compile()
score, loss = model.minevaluate(dataset)
print("accuracy: ", score)
print("loss: ", loss)
'''

# Metodo di valutazione Tony
for f in os.listdir(input_path):
    if f.find(".png") != -1:
        evaluation_img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)

        file_name = f[:f.find(".png")]

        if search_method == "greedy":
            result, _ = sampler.predict_greedy(model, np.array([evaluation_img]))
            print("Result greedy: {}".format(result))
        else:
            beam_width = int(search_method)
            print("Search with beam width: {}".format(beam_width))
            result, _ = sampler.predict_beam_search(model, np.array([evaluation_img]), beam_width=beam_width)
            print("Result beam: {}".format(result))

        with open("{}/{}.gui".format(output_path, file_name), 'w') as out_f:
            out_f.write(result.replace(START_TOKEN, "").replace(END_TOKEN, ""))

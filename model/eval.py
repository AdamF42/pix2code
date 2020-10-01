from __future__ import absolute_import

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import sys

from classes.BeamSearch import *
from classes.Sampler import *
from classes.Utils import *
from classes.Vocabulary import *
from classes.dataset.Generator import *
from classes.model.pix2code import *

argv = sys.argv[1:]

if len(argv) < 2:
    print("Error: not enough argument supplied:")
    print(
        "generate.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>")
    exit(0)
else:
    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    input_path = argv[2]
    print(trained_weights_path)
    print(trained_model_name)
    print(input_path)

meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
input_shape = meta_dataset[0]
output_size = meta_dataset[1]

model = pix2code(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)

dataset = Dataset()
dataset.load(input_path)  # generate_binary_sequences=True)


voc = Vocabulary()
voc.retrieve(path="../bin")

gui_paths, img_paths = Dataset.load_paths_only(input_path)


def get_eval_img(img_path, gui_path):
    evaluation_img = Utils.get_preprocessed_img(img_path, IMAGE_SIZE)
    gui = open(gui_path, 'r')
    token_sequence = [START_TOKEN]
    for line in gui:
        line = line.replace(",", " ,").replace("\n", " \n")
        tokens = line.split(" ")
        for token in tokens:
            token_sequence.append(token)
    token_sequence.append(END_TOKEN)
    return evaluation_img, token_sequence


def predict_greedy(model, input_img, tokens, require_sparse_label=True, sequence_length=150):
    current_context = [voc.vocabulary[PLACEHOLDER]] * (CONTEXT_LENGTH - 1)
    current_context.append(voc.vocabulary[START_TOKEN])
    if require_sparse_label:
        current_context = Utils.sparsify(current_context, output_size)

    predictions = START_TOKEN
    out_probas = []

    score = 0
    total = len(tokens)

    for i in range(0, sequence_length):
        probas = model.predict(input_img, np.array([current_context]))
        prediction = np.argmax(probas)
        out_probas.append(probas)

        new_context = []
        for j in range(1, CONTEXT_LENGTH):
            new_context.append(current_context[j])

        token = voc.token_lookup[prediction]
        if token == tokens[i]:
            score += 1
        sparse_label = voc.binary_vocabulary[token]
        new_context.append(sparse_label)

        current_context = new_context
        predictions += token

        if voc.token_lookup[prediction] == END_TOKEN:
            break

    return predictions, out_probas, score / total * 100


total_score = 0
for i in img_paths:
    gui = i.replace('png', 'gui')
    evaluation_img, tokens = get_eval_img(i, gui)
    _, _, result = predict_greedy(model, np.array([evaluation_img]), tokens[1:-1])
    total_score += result

print("Tony accuracy: ", total_score / len(img_paths))

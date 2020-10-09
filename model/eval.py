from __future__ import absolute_import

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import sys

from classes.BeamSearch import *
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
model = pix2code(input_shape, output_size, trained_weights_path, encoding_type="one_hot")  # TODO: crea parametro
model.load(trained_model_name)

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


def predict_greedy(model, input_img, sequence_length=150):
    current_context = [voc.vocabulary[PLACEHOLDER]] * (CONTEXT_LENGTH - 1)
    current_context.append(voc.vocabulary[START_TOKEN])
    predictions = START_TOKEN
    predicted_tokens = []

    for i in range(0, sequence_length):
        probas = model.predict(input_img, np.array([current_context]))
        prediction = np.argmax(probas)

        new_context = []
        for j in range(1, CONTEXT_LENGTH):
            new_context.append(current_context[j])

        token = voc.token_lookup[prediction]

        predicted_tokens.append(token)

        sparse_label = voc.binary_vocabulary[token]
        new_context.append(sparse_label)

        current_context = new_context
        predictions += token

        if voc.token_lookup[prediction] == END_TOKEN:
            break

    return predicted_tokens


def get_img_score(predicted_tokens, tokens):
    errors = 0
    correct = 0
    if len(tokens) != len(predicted_tokens):
        errors += abs(len(predicted_tokens) - len(tokens))
    for i in range(0, min(len(tokens), len(predicted_tokens))):
        if predicted_tokens[i] != tokens[i]:
            errors += 1
        else:
            correct += 1
    return (correct * 100) / (correct + errors)


total_score = 0
index = 0
for i in img_paths:
    gui = i.replace('png', 'gui')
    evaluation_img, tokens = get_eval_img(i, gui)
    print(gui)
    predicted_tokens = predict_greedy(model, np.array([evaluation_img]))

    result = get_img_score(predicted_tokens, tokens[1:-1])
    total_score += result
    index += 1
    print("Tony accuracy: ", total_score / index)

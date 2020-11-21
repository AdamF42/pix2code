from __future__ import absolute_import

__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import sys

from classes.BeamSearch import *
from classes.Utils import *
from classes.Vocabulary import *
from classes.dataset.Generator import *
from classes.model.pix2code import *

# from model.classes.model.pix2code import *

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
    encoding_type = argv[3]

    print(trained_weights_path)
    print(trained_model_name)
    print(input_path)
    print(encoding_type)

meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)
input_shape = meta_dataset[0]
output_size = meta_dataset[1]
model = pix2code(input_shape, output_size, trained_weights_path, encoding_type=encoding_type)
model.load(trained_model_name)
model.model.summary()

dataset = Dataset()
dataset.load(input_path)  # generate_binary_sequences=True)

voc = Vocabulary()
voc.retrieve(path="../bin")

gui_paths, img_paths = Dataset.load_paths_only(input_path)


# TODO: check if need to do something with whitespaces
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


def create_eval_dict():
    res_eval = {}  # chiave nome immagine, valore lista di stringhe (righe)
    count_eval = 0
    eval_list = []
    for filename in os.listdir(input_path):
        if filename.endswith(".gui"):
            count_eval += 1
            eval_list.append(filename)
            with open(os.path.join(input_path, f"{filename}"), 'r') as img:

                lines = []
                for el in img.readlines():
                    line = el.replace(" ", "  ") \
                        .replace(",", " ,") \
                        .replace("\n", " \n")
                    tokens = line.split(" ")
                    tokens = map(lambda x: " " if x == "" else x, tokens)
                    tokens = filter(lambda x: False if x == " " else True, tokens)
                    for token in tokens:
                        lines.append(token)

            res_eval[filename] = lines
    assert (count_eval == 250)
    return eval_list, res_eval


def create_code_dict():
    res_code = {}  # chiave nome immagine, valore lista di stringhe (righe)
    count_code = 0
    code_list = []
    for i in os.listdir(input_path):
        if i.endswith(".gui"):
            count_code += 1
            code_list.append(i)
            img_path = input_path+i.replace("gui","png")
            evaluation_img = Utils.get_preprocessed_img(img_path, IMAGE_SIZE)
            tokens = predict_greedy(model, np.array([evaluation_img]))
            res_code[i] = tokens
    assert (count_code == 250)
    return code_list, res_code


def check_existing_el(code_list, eval_list):
    result = all(elem in eval_list for elem in code_list)
    if result:
        print("Yes, eval_list contains all elements in code_list")
    else:
        print("No, eval_list does not contains all elements in code_list")


def compare(eval_el, code_el):
    correct = 0
    error = 0
    for i in range(0, min(len(eval_el), len(code_el))):
        if eval_el[i] == code_el[i]:
            correct += 1
        else:
            error += 1
    tot = correct + error
    return correct, error, tot


def print_accuracy(len_difference, tot_correct, tot_error, tot_tot):
    print("CORRETTI: ", tot_correct)
    print("ERRATI: ", tot_error + len_difference)
    print("TOTALI: ", tot_tot + len_difference)
    tot_correct_percentuale = (tot_correct / (tot_tot + len_difference)) * 100
    tot_error_percentuale = ((tot_error + len_difference) / (tot_tot + len_difference)) * 100
    print("PERCENTUALE CORRETTI: ", tot_correct_percentuale)
    print("PERCENTUALE ERRATI: ", tot_error_percentuale)
    assert round(tot_correct_percentuale, 2) + round(tot_error_percentuale, 2) == 100.0


tot_correct = 0
tot_error = 0
tot_tot = 0
len_difference = 0

eval_list, res_eval = create_eval_dict()

code_list, res_code = create_code_dict()

check_existing_el(code_list, eval_list)

for key in res_eval:
    if len(res_code[key]) != len(res_eval[key]):
        # se ho lunghezze diverse conto come errore la loro differenza
        len_difference += abs(len(res_code[key]) - len(res_eval[key]))
    corr, err, tot = compare(res_eval[key], res_code[key])
    tot_correct += corr
    tot_error += err
    tot_tot += tot

print_accuracy(len_difference, tot_correct, tot_error, tot_tot)

import os
from functools import reduce

import gensim
from gensim.models import Word2Vec
from tensorflow.python.keras.models import load_model

from w2v_test.dataset.utils import get_token_sequences_with_max_seq_len, get_token_from_gui
from w2v_test.models.pix2code_w2v_embedding import Pix2codeW2VEmbedding

IMG_W2V_TRAIN_DIR = '/home/adamf42/Projects/pix2code/datasets/web/prove'
IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/prove'

print("################################## GENSIM ##################################")

print('\nPreparing the sentences...')

tokens = get_token_sequences_with_max_seq_len(IMG_W2V_TRAIN_DIR, [])

max_sentence_len = tokens['max_sentence_len']
sentences = tokens['sentences']

print("MAX SENTENCE LENGHT: " + str(max_sentence_len))
print("NUMBER OF SENTENCIES: " + str(len(sentences)))

print(sentences)

print('\nLoad word2vec...')
word_model: Word2Vec = gensim.models.Word2Vec.load('/home/adamf42/Projects/pix2code/w2v_test/word2vec.model')
pretrained_weights = word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape

print("################################## MODEL ##################################")

new_model_rms = Pix2codeW2VEmbedding(pretrained_weights=pretrained_weights)

new_model_rms.compile()

new_model_rms: Pix2codeW2VEmbedding = load_model('/home/adamf42/Projects/pix2code/w2v_test/pix2code_new_w2v')

print("################################## EVAL ##################################")


# image_to_predict = '/home/adamf42/Projects/pix2code/datasets/web/single/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.png'

# prediction = Pix2codeW2VEmbedding.predict_image(new_model, image_to_predict, word_model, max_sentence_len)

# print(prediction)


score_dict = {}


def create_eval_dict():
    res_eval = {}  # chiave nome immagine, valore lista di stringhe (righe)
    eval_list = []
    for filename in os.listdir(IMG_PATH):
        if not filename.endswith(".gui"):
            continue
        gui = open(f'{IMG_PATH}/{filename}', 'r')
        token_sequences = get_token_from_gui(gui)
        res_eval[filename] = token_sequences
        eval_list.append(filename)

    return eval_list, res_eval


def create_code_dict():
    res_code = {}  # chiave nome immagine, valore lista di stringhe (righe)
    code_list = []
    for i in os.listdir(IMG_PATH):
        if i.endswith(".png"):
            tokens = Pix2codeW2VEmbedding.predict_image(new_model_rms, f'{IMG_PATH}/{i}', word_model, max_sentence_len)
            print(i)
            print(reduce(lambda a, b: f"{a} {b}", tokens))
            res_code[i.replace("png", "gui")] = tokens
            code_list.append(i.replace("png", "gui"))
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
            # print(f'actual: {eval_el[i]} predicted: {code_el[i]}')
            update_score(eval_el[i], code_el[i])
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


def update_score(actual_token, predicted_token):
    if (actual_token in score_dict):
        score_dict[actual_token].append(predicted_token)
    else:
        score_dict[actual_token] = [predicted_token]


def elaborate_score():
    for key, value in score_dict.items():
        dict = {}
        for v in value:
            if v in dict:
                dict[v] = dict[v] + 1
            else:
                dict[v] = 1
        score_dict[key] = dict


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

elaborate_score()

print(score_dict)
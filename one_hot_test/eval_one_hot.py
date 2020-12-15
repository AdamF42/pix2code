import os

from tensorflow.python.keras.models import load_model
from one_hot_test.dataset.Dataset import Dataset
from one_hot_test.models.pix2code_one_hot_embedding import Pix2codeOneHotEmbedding
from w2v_test.dataset.utils import get_token_sequences_with_max_seq_len, get_token_from_gui

IMG_ONEHOT_TRAIN_DIR = '../datasets/web/training_set'
IMG_PATH = '../datasets/web/mini_eval'  # eval_set

print("################################## DATASET ##################################")

print('\nPreparing the sentences...')

tokens = get_token_sequences_with_max_seq_len(IMG_ONEHOT_TRAIN_DIR)

max_sentence_len = tokens['max_sentence_len']
sentences = tokens['sentences']

print("MAX SENTENCE LENGHT: " + str(max_sentence_len))
print("NUMBER OF SENTENCIES: " + str(len(sentences)))

print('\nCreate one_hot encoding...')
dataset = Dataset()
for sentence in tokens['sentences']:
    for word in sentence:
        dataset.voc.append(word)
dataset.load_with_one_hot_encoding(IMG_ONEHOT_TRAIN_DIR)

print("voc: ", dataset.voc.vocabulary)
print("binary voc: ", dataset.voc.binary_vocabulary)
print("emb matrix: ", dataset.voc.embedding_matrix)

vocab_size = dataset.voc.size

print('Result embedding shape:', vocab_size)  # 18

print("################################## MODEL ##################################")

new_model = Pix2codeOneHotEmbedding(dataset.voc.embedding_matrix)

new_model.compile()

new_model = load_model("pix2code_new_one_hot_RMS")


# shape = [(None, 256, 256, 3), (None, 48)]
# inputs = tf.keras.Input(shape=shape)
# new_model.call(inputs)

print("################################## EVAL ##################################")

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
            tokens = Pix2codeOneHotEmbedding.predict_image(new_model, f'{IMG_PATH}/{i}', dataset.voc, max_sentence_len)
            res_code[i.replace("png","gui")] = tokens
            code_list.append(i.replace("png","gui"))
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


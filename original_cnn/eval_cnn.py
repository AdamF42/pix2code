import glob
import os
from functools import reduce

import gensim
import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.keras.models import load_model

from original_cnn.Pix2CodeOriginalCnnModel import Pix2CodeOriginalCnnModel
from original_cnn.costants import TOKENS_TO_INDEX
from original_cnn.generator import DataGenerator
from w2v_test.dataset.dataset import Dataset
from w2v_test.dataset.utils import get_token_sequences_with_max_seq_len, get_token_from_gui, get_preprocessed_img
from w2v_test.models.pix2code_w2v_embedding import Pix2codeW2VEmbedding

# IMG_W2V_TRAIN_DIR = '/home/adamf42/Projects/pix2code/datasets/web/eval_set'
IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/prove'

print("################################## MODEL ##################################")

voc = list(TOKENS_TO_INDEX.keys())
output_names = map(lambda x: "open_bracket" if x == "{" else x, voc)
output_names = map(lambda x: "close_bracket" if x == "}" else x, output_names)
output_names = list(map(lambda x: "comma" if x == "," else x, output_names))

new_model = Pix2CodeOriginalCnnModel(output_names)

new_model.compile()

# new_model: Pix2CodeOriginalCnnModel = load_model('pix2code_cnn_china')
new_model: Pix2CodeOriginalCnnModel = load_model('pix2code_cnn')

print("################################## EVAL ##################################")

# infinite loop
labels, img_paths = Dataset.load_paths_only(IMG_PATH)

generator = DataGenerator(img_paths, labels, output_names, batch_size=1)

images_to_predict = [generator.__getitem__(i) for i in range(len(img_paths))]

# images_to_predict = ['/home/adamf42/Projects/pix2code/datasets/web/eval_set/0EBD0467-076F-4946-8549-C3EEF47F37AF.png']


# images_to_predict = list(glob.glob(os.path.join(IMG_PATH, "*.npz")))
#
# images_to_predict = {'img_data': np.array([get_preprocessed_img(img) for img in images_to_predict])}
#
# print(images_to_predict['img_data'].shape)
#
# prediction = new_model.predict(images_to_predict)
#
# print(prediction)


# score_dict = {}
#
#
# def create_eval_dict():
#     res_eval = {}  # chiave nome immagine, valore lista di stringhe (righe)
#     eval_list = []
#     for filename in os.listdir(IMG_PATH):
#         if not filename.endswith(".gui"):
#             continue
#         gui = open(f'{IMG_PATH}/{filename}', 'r')
#         token_sequences = get_token_from_gui(gui)
#         res_eval[filename] = token_sequences
#         eval_list.append(filename)
#
#     return eval_list, res_eval
#
#
# def create_code_dict():
#     res_code = {}  # chiave nome immagine, valore lista di stringhe (righe)
#     code_list = []
#     for i in os.listdir(IMG_PATH):
#         if i.endswith(".png"):
#             tokens = Pix2codeW2VEmbedding.predict_image(new_model_rms, f'{IMG_PATH}/{i}', word_model, max_sentence_len)
#             print(i)
#             print(reduce(lambda a, b: f"{a} {b}", tokens))
#             res_code[i.replace("png", "gui")] = tokens
#             code_list.append(i.replace("png", "gui"))
#     return code_list, res_code
#
#
# def check_existing_el(code_list, eval_list):
#     result = all(elem in eval_list for elem in code_list)
#     if result:
#         print("Yes, eval_list contains all elements in code_list")
#     else:
#         print("No, eval_list does not contains all elements in code_list")
#
#
# def compare(eval_el, code_el):
#     correct = 0
#     error = 0
#     for i in range(0, min(len(eval_el), len(code_el))):
#         if eval_el[i] == code_el[i]:
#             correct += 1
#         else:
#             # print(f'actual: {eval_el[i]} predicted: {code_el[i]}')
#             update_score(eval_el[i], code_el[i])
#             error += 1
#     tot = correct + error
#     return correct, error, tot
#
#
# def print_accuracy(len_difference, tot_correct, tot_error, tot_tot):
#     print("CORRETTI: ", tot_correct)
#     print("ERRATI: ", tot_error + len_difference)
#     print("TOTALI: ", tot_tot + len_difference)
#     tot_correct_percentuale = (tot_correct / (tot_tot + len_difference)) * 100
#     tot_error_percentuale = ((tot_error + len_difference) / (tot_tot + len_difference)) * 100
#     print("PERCENTUALE CORRETTI: ", tot_correct_percentuale)
#     print("PERCENTUALE ERRATI: ", tot_error_percentuale)
#     assert round(tot_correct_percentuale, 2) + round(tot_error_percentuale, 2) == 100.0
#
#
# def update_score(actual_token, predicted_token):
#     if (actual_token in score_dict):
#         score_dict[actual_token].append(predicted_token)
#     else:
#         score_dict[actual_token] = [predicted_token]
#
#
# def elaborate_score():
#     for key, value in score_dict.items():
#         dict = {}
#         for v in value:
#             if v in dict:
#                 dict[v] = dict[v] + 1
#             else:
#                 dict[v] = 1
#         score_dict[key] = dict
#
#
# tot_correct = 0
# tot_error = 0
# tot_tot = 0
# len_difference = 0
#
# eval_list, res_eval = create_eval_dict()
#
# code_list, res_code = create_code_dict()
#
# check_existing_el(code_list, eval_list)
#
# for key in res_eval:
#     if len(res_code[key]) != len(res_eval[key]):
#         # se ho lunghezze diverse conto come errore la loro differenza
#         len_difference += abs(len(res_code[key]) - len(res_eval[key]))
#     corr, err, tot = compare(res_eval[key], res_code[key])
#     tot_correct += corr
#     tot_error += err
#     tot_tot += tot
#
# print_accuracy(len_difference, tot_correct, tot_error, tot_tot)
#
# elaborate_score()
#
# print(score_dict)
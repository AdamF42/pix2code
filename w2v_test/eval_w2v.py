import gensim
import numpy as np
from gensim.models import Word2Vec

from cnn.VocabularyOneHot import VocabularyOneHot
from cnn.generator import DataGenerator
from utils.costants import PLACEHOLDER, COMMA, START_TOKEN, END_TOKEN, TOKEN_TO_EXCLUDE
from utils.dataset import Dataset
from utils.utils import load_pickle, get_output_names, eval_code_error
from w2v_test.models.VocabularyW2V import VocabularyW2V
from w2v_test.models.W2VCnnModel import W2VCnnModel

# IMG_W2V_TRAIN_DIR = '/home/adamf42/Projects/pix2code/datasets/web/single'
# IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/single'

# IMG_PATH_TRAIN = '../datasets/web/train_features'
IMG_PATH_VALIDATION = '../datasets/web/validation_features'
# IMG_PATH_TEST = '../datasets/web/test_features'

print("################################## GENSIM ##################################")

print('\nPreparing the sentences...')

# # tokens_sequences = get_token_sequences_with_max_seq_len(IMG_W2V_TRAIN_DIR, is_with_output_name=True)
# tokens_sequences = load_pickle('../pickle/tokens_sequences_no_spaces.pickle')
# # save_pickle(tokens_sequences, '../pickle/tokens_sequences_no_spaces.pickle')
# max_sentence_len = tokens_sequences['max_sentence_len']
# sentences = tokens_sequences['sentences']
# sentences.append([PLACEHOLDER])
# print("MAX SENTENCE LENGHT: " + str(max_sentence_len))
# print("NUMBER OF SENTENCIES: " + str(len(sentences)))

print('\nLoad word2vec...')
# word_model: Word2Vec = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=5, iter=400)
word_model: Word2Vec = gensim.models.Word2Vec.load('../instances/word2vec_no_spaces_output_name.model')
# word_model.save('../instances/word2vec_no_spaces_output_name.model')
pretrained_weights = word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print("emdedding_size: {}, vocab_size: {}".format(emdedding_size, vocab_size))
print("vocab: {}".format(word_model.wv.vocab.keys()))

voc = VocabularyW2V(word_model)
words = load_pickle('../pickle/output_names.pickle')
words = words + [COMMA, START_TOKEN, END_TOKEN, PLACEHOLDER]
output_names = get_output_names(words)
tokens_sequences = load_pickle('../pickle/tokens_sequences_no_spaces.pickle')
max_sentence_len = tokens_sequences['max_sentence_len']
tokens_to_exclude = TOKEN_TO_EXCLUDE

mapping = load_pickle('../pickle/one_hot_mapping.pickle')
voc_one_hot = VocabularyOneHot(mapping)

def code_encoder_one_hot(sequence):
    encoded_sequences = [mapping[i] for i in sequence]
    code_one_hot = np.array(encoded_sequences)
    pad = np.tile(mapping[PLACEHOLDER], (max_sentence_len - len(code_one_hot), 1))
    return np.concatenate([code_one_hot,pad])

print("################################## MODEL ##################################")

# new_model_rms = Pix2codeW2VEmbedding(pretrained_weights=pretrained_weights)
#
# new_model_rms.compile()
#
# new_model_rms: Pix2codeW2VEmbedding = load_model('/home/adamf42/Projects/pix2code/w2v_test/pix2code_new_w2v')
validation_labels, validation_img_paths = Dataset.load_paths_only(IMG_PATH_VALIDATION)

cnn_samples_validation_features = load_pickle('../pickle/cnn_image_samples_validation_features.pickle')

cnn_validation_generator_data = DataGenerator(validation_img_paths, validation_labels, output_names, tokens_to_exclude,
                                              samples=cnn_samples_validation_features, batch_size=1)

# cnn_validation_generator_data = DataGenerator(validation_img_paths, validation_labels, output_names, tokens_to_exclude,
#                                               code_encoder=code_encoder_one_hot, batch_size=1)

# cnn_samples_test_features_data = DataGenerator(test_img_paths, test_labels, output_names, tokens_to_exclude,
#                                                samples=cnn_samples_test_features, batch_size=1)

test_data = list(map(lambda x: (x[0]['img_data'],
                                list(filter(lambda x: x != PLACEHOLDER,
                                            [voc_one_hot.index_to_word(i) for i in np.argmax(x[1]['code'][0], axis=1)]))),
                     [cnn_validation_generator_data.__getitem__(i) for i in range(len(validation_img_paths))]))
test_data = [(img, [voc.word_to_index(elem) for elem in code]) for img, code in test_data]

model_instance = W2VCnnModel(w2v_pretrained_weights=pretrained_weights,
                             words=words,
                             image_count_words=voc.get_tokens(),
                             max_code_length=max_sentence_len,
                             dropout_ratio=0.1)
image = '../datasets/web/prove/0D1C8ADB-D9F0-48EC-B5AA-205BCF96094E.png'
# img_build, _ = cnn_validation_generator_data.__getitem__(0)
model_instance.compile()
# img_build = img_build['img_data'][0]
model_instance.predict_image(image, voc)
model_save_path = '../instances/W2VCnnModel.h5'

print("################################## EVAL ##################################")
train_df = eval_code_error(model_instance, test_data, validation_img_paths, voc, max_sentence_len)

# print(train_df.head())
print("MEAN ERROR: " + str(train_df.error.mean()))
print("Correct predictions: " + str(train_df.correctly_predicted.mean()))
print("same length: " + str(train_df.same_length.mean()))
print("active_button_correct: " + str(train_df.active_button_correct.mean()))
print("button_color_correct: " + str(train_df.button_color_correct.mean()))

# image_to_predict = '/home/adamf42/Projects/pix2code/datasets/web/single/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.png'

# prediction = Pix2codeW2VEmbedding.predict_image(new_model, image_to_predict, word_model, max_sentence_len)

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

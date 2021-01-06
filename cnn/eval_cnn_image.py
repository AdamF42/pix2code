import numpy as np
import pandas

from cnn.CnnImageModel import CnnImageModel
from cnn.VocabularyOneHot import VocabularyOneHot
from cnn.generator import DataGenerator
from utils.costants import TOKEN_TO_EXCLUDE, COMMA, START_TOKEN, END_TOKEN, PLACEHOLDER
from utils.dataset import Dataset
from utils.utils import load_pickle, get_output_names, eval_code_error

pandas.set_option("display.max_rows", None, "display.max_columns", None)

IMG_PATH_TRAIN = '../datasets/web/train_features'
IMG_PATH_VALIDATION = '../datasets/web/validation_features'
IMG_PATH_TEST = '../datasets/web/test_features'

words = load_pickle('../pickle/output_names.pickle')
words = words + [COMMA, START_TOKEN, END_TOKEN, PLACEHOLDER]
output_names = get_output_names(words)
tokens_sequences = load_pickle('../pickle/tokens_sequences_no_spaces.pickle')
max_sentence_len = tokens_sequences['max_sentence_len']
tokens_to_exclude = TOKEN_TO_EXCLUDE

print("################################## DATA ###################################")

mapping = load_pickle('../pickle/one_hot_mapping.pickle')
voc = VocabularyOneHot(mapping)

train_labels, train_paths = Dataset.load_paths_only(IMG_PATH_TRAIN)
validation_labels, validation_img_paths = Dataset.load_paths_only(IMG_PATH_VALIDATION)
test_labels, test_img_paths = Dataset.load_paths_only(IMG_PATH_TEST)

cnn_samples_train_features = load_pickle('../pickle/cnn_image_samples_train_features.pickle')
cnn_samples_validation_features = load_pickle('../pickle/cnn_image_samples_validation_features.pickle')
cnn_samples_test_features = load_pickle('../pickle/cnn_image_samples_test_features.pickle')

cnn_train_generator_data = DataGenerator(train_paths, train_labels, output_names, tokens_to_exclude,
                                         samples=cnn_samples_train_features, batch_size=1)

cnn_validation_generator_data = DataGenerator(validation_img_paths, validation_labels, output_names, tokens_to_exclude,
                                              samples=cnn_samples_validation_features, batch_size=1)

cnn_samples_test_features_data = DataGenerator(test_img_paths, test_labels, output_names, tokens_to_exclude,
                                               samples=cnn_samples_test_features, batch_size=1)

test_data = list(map(lambda x: (x[0]['img_data'], [i for i in np.argmax(x[1]['code'][0], axis=1)]),
                     [cnn_samples_test_features_data.__getitem__(i) for i in range(len(test_img_paths))]))

print("################################## MODEL ##################################")
img_build, label = cnn_train_generator_data.__getitem__(0)

# new_model = CnnModel(output_names, image_out=True, dense_layer_size=512, dropout_ratio=0.1)
new_model = CnnImageModel(words, output_names, max_sentence_length=max_sentence_len,
                          dense_layer_size=512, dropout_ratio=0.1)
new_model.compile()
new_model.predict(img_build)
# new_model.load_weights('../instances/best_model_cnn_final.h5')
new_model.load_weights('../instances/CnnImageModel.h5')
# test = new_model.predict_image(img_build['img_data'], voc)

# print(test)

print("################################## EVAL ##################################")
train_df = eval_code_error(new_model, test_data, test_img_paths, voc, max_sentence_len)

# print(train_df.head())
print("MEAN ERROR: " + str(train_df.error.mean()))
print("Correct predictions: " + str(train_df.correctly_predicted.mean()))
print("same length: " + str(train_df.same_length.mean()))
print("active_button_correct: " + str(train_df.active_button_correct.mean()))
print("button_color_correct: " + str(train_df.button_color_correct.mean()))

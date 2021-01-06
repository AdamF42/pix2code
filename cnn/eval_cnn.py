import pandas

from cnn.CnnImageModel import CnnImageModel
from cnn.CnnModel import CnnModel
from cnn.CnnModelOriginal import CnnModelOriginal
from cnn.generator import DataGenerator
from utils.utils import load_pickle, eval_cnn_model, get_output_names
from utils.costants import TOKEN_TO_EXCLUDE, COMMA, START_TOKEN, END_TOKEN, PLACEHOLDER
from utils.dataset import Dataset

pandas.set_option("display.max_rows", None, "display.max_columns", None)

IMG_PATH_TRAIN = '../datasets/web/train_features'
IMG_PATH_TRAIN_AUGMENTED = '../datasets/web/augmented_train_features'
IMG_PATH_VALIDATION = '../datasets/web/validation_features'
IMG_PATH_TEST = '../datasets/web/test_features'

words = load_pickle('../pickle/output_names.pickle')
words = words + [COMMA, START_TOKEN, END_TOKEN, PLACEHOLDER]
output_names = get_output_names(words)
tokens_sequences = load_pickle('../pickle/tokens_sequences_no_spaces.pickle')
max_sentence_len = tokens_sequences['max_sentence_len']
tokens_to_exclude = TOKEN_TO_EXCLUDE

print("################################## DATA ###################################")

train_labels, train_paths = Dataset.load_paths_only(IMG_PATH_TRAIN)
augmented_train_labels, augmented_train_img_paths = Dataset.load_paths_only(IMG_PATH_TRAIN_AUGMENTED)
validation_labels, validation_img_paths = Dataset.load_paths_only(IMG_PATH_VALIDATION)
test_labels, test_img_paths = Dataset.load_paths_only(IMG_PATH_TEST)


cnn_samples_train_features = load_pickle('../pickle/cnn_samples_train_features.pickle')
cnn_samples_agumented_train_features = load_pickle('../pickle/cnn_samples_agumented_train_features.pickle')
cnn_samples_validation_features = load_pickle('../pickle/cnn_samples_validation_features.pickle')
cnn_samples_test_features = load_pickle('../pickle/cnn_samples_test_features.pickle')

cnn_train_generator_data = DataGenerator(train_paths, train_labels, output_names, tokens_to_exclude,
                          samples=cnn_samples_train_features, batch_size=1)

cnn_agumented_train_generator_data = DataGenerator(augmented_train_img_paths, augmented_train_labels, output_names, tokens_to_exclude,
                          samples=cnn_samples_agumented_train_features, batch_size=1)

cnn_validation_generator_data = DataGenerator(validation_img_paths, validation_labels, output_names, tokens_to_exclude,
                               samples=cnn_samples_validation_features, batch_size=1)

cnn_samples_test_features_data = DataGenerator(test_img_paths, test_labels, output_names, tokens_to_exclude,
                               samples=cnn_samples_test_features, batch_size=1)

train_data = [cnn_train_generator_data.__getitem__(i) + ({'name': train_paths[i]},) for i in range(len(train_paths))]
agumented_train_data = [cnn_agumented_train_generator_data.__getitem__(i) + ({'name': augmented_train_img_paths[i]},) for i in range(len(augmented_train_img_paths))]
validation_data = [cnn_validation_generator_data.__getitem__(i) + ({'name': validation_img_paths[i]},) for i in range(len(validation_img_paths))]
test_data = [cnn_samples_test_features_data.__getitem__(i) + ({'name': test_img_paths[i]},) for i in range(len(test_img_paths))]




print("################################## MODEL ##################################")
print(len(output_names))
# new_model = CnnModel(output_names, image_out=True, dense_layer_size=512, dropout_ratio=0.1)
new_model = CnnModel(output_names, dense_layer_size=512, dropout_ratio=0.1)
# new_model = CnnImageModel(words, output_names,  max_sentence_length=max_sentence_len,
#                           dense_layer_size=512, dropout_ratio=0.1)
new_model.compile()
img_build, label = cnn_train_generator_data.__getitem__(0)
# print(list(label.keys()))

new_model.predict(img_build)
# new_model.load_weights('../instances/best_model_cnn_final.h5')
new_model.load_weights('../instances/CnnModel4.h5')

print("################################## EVAL ##################################")
train_errors, train_y, train_predictions = eval_cnn_model(new_model, train_data, output_names)
print('TRAIN')
print(train_errors.head())
train_errors, train_y, train_predictions = eval_cnn_model(new_model, agumented_train_data, output_names)
print('AUGUMENTED TRAIN')
print(train_errors.head())
train_errors, train_y, train_predictions = eval_cnn_model(new_model, validation_data, output_names)
print('VALIDATION')
print(train_errors.head())
train_errors, train_y, train_predictions = eval_cnn_model(new_model, test_data, output_names)
print('TEST')
print(train_errors)

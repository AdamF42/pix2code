import numpy as np

from cnn.CnnImageModel import CnnImageModel
from cnn.CnnModel import CnnModel
from cnn.CnnModelOriginal import CnnModelOriginal
from cnn.VocabularyOneHot import VocabularyOneHot
from cnn.generator import DataGenerator
from utils.utils import load_pickle, eval_cnn_model, get_output_names, eval_code_error
from utils.costants import TOKEN_TO_EXCLUDE, COMMA, START_TOKEN, END_TOKEN, PLACEHOLDER
from utils.dataset import Dataset

# IMG_PATH = '../datasets/web/train_features'
IMG_PATH = '../datasets/web/validation_features'

words = load_pickle('../pickle/output_names.pickle')
words = words + [COMMA, START_TOKEN, END_TOKEN, PLACEHOLDER]
output_names = get_output_names(words)
tokens_sequences = load_pickle('../pickle/tokens_sequences_no_spaces.pickle')
max_sentence_len = tokens_sequences['max_sentence_len']
tokens_to_exclude = TOKEN_TO_EXCLUDE

print("################################## DATA ###################################")

labels, img_paths = Dataset.load_paths_only(IMG_PATH)
# train_samples = load_pickle('../pickle/image_model_train_generator_samples.pickle')
train_samples = load_pickle('../pickle/image_model_val_generator_samples.pickle')

mapping = load_pickle('../pickle/one_hot_mapping.pickle')

voc = VocabularyOneHot(mapping)

generator = DataGenerator(img_paths, labels, output_names, tokens_to_exclude, samples=train_samples, batch_size=1)

# data = [generator.__getitem__(i) + ({'name': img_paths[i]},) for i in range(len(img_paths))]

# def get_word(list):


data = list(map(lambda x: (x[0]['img_data'], [i for i in np.argmax(x[1]['code'][0], axis=1)]),
                [ generator.__getitem__(i) for i in range(len(img_paths))]))

print("################################## MODEL ##################################")
img_build, label = generator.__getitem__(0)

# new_model = CnnModel(output_names, image_out=True, dense_layer_size=512, dropout_ratio=0.1)
new_model = CnnImageModel(words, output_names,  max_sentence_length=max_sentence_len,
                          dense_layer_size=512, dropout_ratio=0.1)
new_model.compile()
new_model.predict(img_build)
# new_model.load_weights('../instances/best_model_cnn_final.h5')
new_model.load_weights('../instances/CnnImageModel.h5')
# test = new_model.predict_image(img_build['img_data'], voc)

# print(test)

print("################################## EVAL ##################################")
# def eval_code_error(model_instance, data, data_paths, voc:Vocabulary, max_sentence_len, index_value='accuracy'):
train_df = eval_code_error(new_model, data, img_paths, voc, max_sentence_len)

# print(train_df.head())
print(train_df.error.mean())
print(train_df.correctly_predicted.mean())
print(train_df.same_length.mean())
print(train_df.active_button_correct.mean())
print(train_df.button_color_correct.mean())
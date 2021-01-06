from typing import Collection

import gensim
import numpy as np
from gensim.models import Word2Vec
from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite
from tensorflow.keras.optimizers import RMSprop

from cnn.CnnImageModel import CnnImageModel
from cnn.generator import DataGenerator
from utils.costants import TOKEN_TO_EXCLUDE, END_TOKEN, START_TOKEN, COMMA, PLACEHOLDER
from utils.dataset import Dataset
from utils.utils import load_pickle, get_output_names, save_pickle
from w2v_test.models.VocabularyW2V import VocabularyW2V

IMG_PATH = '../datasets/web/train_features'
IMG_PATH_EVAL = '../datasets/web/validation_features'
IMG_PATH_TEST = '../datasets/web/test_features'

words = load_pickle('../pickle/output_names.pickle')
words = words + [COMMA, START_TOKEN, END_TOKEN, PLACEHOLDER]
output_names = get_output_names(words)
tokens_sequences = load_pickle('../pickle/tokens_sequences_no_spaces.pickle')
max_sentence_len = tokens_sequences['max_sentence_len']
tokens_to_exclude = TOKEN_TO_EXCLUDE

# word_model: Word2Vec = gensim.models.Word2Vec.load('../instances/word2vec_no_spaces_output_name.model')
# word_model.save('../instances/word2vec_no_spaces_output_name.model')
# print('Result embedding shape:', pretrained_weights.shape)
# print("emdedding_size: {}, vocab_size: {}".format(emdedding_size, vocab_size))
# print("vocab: {}".format(word_model.wv.vocab.keys()))

# voc = VocabularyW2V(word_model)


# def code_encoder_w2v(sequence):
#     code_w2v = np.array(voc.w2v_encode(sequence))
#     pad = np.tile(voc.w2v_encode([PLACEHOLDER]), (max_sentence_len - len(code_w2v), 1))
#     return np.concatenate([code_w2v,pad])

# tokens = voc.get_tokens()
# sequence_to_numbers = [i for i in range(len(tokens))]
# sequence_to_numbers = to_categorical(sequence_to_numbers)
# mapping = {token: i for i, token in zip(sequence_to_numbers, tokens)}

# save_pickle(mapping, '../pickle/one_hot_mapping.pickle')

mapping = load_pickle('../pickle/one_hot_mapping.pickle')
#
def code_encoder_one_hot(sequence):
    encoded_sequences = [mapping[i] for i in sequence]
    code_one_hot = np.array(encoded_sequences)
    pad = np.tile(mapping[PLACEHOLDER], (max_sentence_len - len(code_one_hot), 1))
    return np.concatenate([code_one_hot,pad])


print("################################# GENERATORS ##################################")

def save_samples(features_path: Collection[str], base_name):
    for path in features_path:
        labels, img_paths = Dataset.load_paths_only(path)
        build_generator = DataGenerator(img_paths,labels, output_names, tokens_to_exclude,
                          code_encoder=code_encoder_one_hot, batch_size=32)
        save_pickle(build_generator.samples, f'../pickle/{base_name}_{path.split("/")[-1]}.pickle')

# labels, img_paths = Dataset.load_paths_only(IMG_PATH)
# labels_eval, img_paths_eval = Dataset.load_paths_only(IMG_PATH_EVAL)

save_samples([IMG_PATH, IMG_PATH_EVAL, IMG_PATH_TEST],'cnn_image_samples')

# generator_samples = load_pickle('../pickle/train_features.pickle')
# generator_eval_samples = load_pickle('../pickle/val_features.pickle')

# generator = DataGenerator(img_paths,labels, output_names, tokens_to_exclude,
#                           code_encoder=code_encoder_one_hot, batch_size=32)
# generator_val = DataGenerator(img_paths_eval, labels_eval, output_names, tokens_to_exclude,
#                                code_encoder=code_encoder_one_hot, batch_size=32)

# save_pickle(generator.samples, '../pickle/image_model_train_generator_samples.pickle')
# save_pickle(generator_val.samples, '../pickle/image_model_val_generator_samples.pickle')

# train_samples = load_pickle('../pickle/image_model_train_generator_samples.pickle')
# eval_samples = load_pickle('../pickle/image_model_val_generator_samples.pickle')

# build_generator = DataGenerator(img_paths_eval, labels_eval, output_names, tokens_to_exclude,
#                                                       samples=eval_samples, batch_size=1)
#
# cnn_image_train_generator = iter_sequence_infinite(DataGenerator(img_paths, labels, output_names, tokens_to_exclude,
#                                                                  samples=train_samples, batch_size=32))
# cnn_image_val_generator = iter_sequence_infinite(DataGenerator(img_paths_eval, labels_eval, output_names, tokens_to_exclude,
#                                                                samples=eval_samples, batch_size=32))
#
# # save_pickle(generator.samples, '../pickle/train_generator_samples.pickle')
# # save_pickle(generator_eval.samples, '../pickle/eval_generator_samples.pickle')
#
# print("################################## MODEL ######################################")
#
# model_instance = CnnImageModel(words, output_names, max_sentence_length=max_sentence_len,
#                           dense_layer_size=512, dropout_ratio=0.1)
#
# img_build, labels = build_generator.__getitem__(0)
# test = model_instance.predict(img_build)
# model_instance.compile()
#
# # Do the transfer learning
# transfer_learning_model_save_path = '../instances/CnnModel.h5'
# model_instance.load_weights(transfer_learning_model_save_path,by_name=True)
#
# # Do initial training with transferred layers locked
# for layer_name in ['cnn_unit','counter_unit']:
#     layer = model_instance.get_layer(layer_name)
#     layer.trainable = False
# loss = {word+"_count": 'mse' for word in model_instance.image_count_words}
# loss.update({'code':'categorical_crossentropy'})
# loss_weights = {word+"_count": 1/len( model_instance.image_count_words) for word in model_instance.image_count_words}
# loss_weights.update({'code':1.0})
# model_instance.compile(loss=loss, loss_weights=loss_weights,optimizer=RMSprop(lr=0.0001, clipvalue=1.0))
#
#
#
#
# print("################################## TRAIN ######################################")
#
# # checkpoint_filepath = '/content/gdrive/MyDrive/Colab Notebooks/CnnImageModel.h5'
# checkpoint_filepath = '../instances/CnnImageModel.h5'
#
# import tensorflow as tf
#
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     save_best_only=True)
#
# early_stopping = EarlyStopping(
#     restore_best_weights=True,
#     patience=10)
#
# history = model_instance.fit(cnn_image_train_generator,
#                              validation_data=cnn_image_val_generator,
#                              epochs=10000,
#                              steps_per_epoch=int(len(img_paths) / 32) * 8,
#                              validation_steps=int(len(img_paths_eval) / 32),
#                              callbacks=[early_stopping])

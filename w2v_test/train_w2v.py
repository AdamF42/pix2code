import gensim
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite
from tensorflow.keras.optimizers import RMSprop
from utils.costants import PLACEHOLDER, COMMA, START_TOKEN, END_TOKEN, TOKEN_TO_EXCLUDE
from utils.dataset import Dataset
from utils.utils import load_pickle
from w2v_test.generator.generator import DataGenerator
from w2v_test.models.VocabularyW2V import VocabularyW2V
from w2v_test.models.W2VCnnModel import W2VCnnModel

IMG_W2V_TRAIN_DIR = '../datasets/web/all_data'
IMG_PATH = '../datasets/web/train_features'
IMG_PATH_EVAL = '../datasets/web/validation_features'

print("################################## GENSIM ##################################")

print('\nPreparing the sentences...')

# tokens_sequences = get_token_sequences_with_max_seq_len(IMG_W2V_TRAIN_DIR, is_with_output_name=True)
tokens_sequences = load_pickle('../pickle/tokens_sequences_no_spaces.pickle')
# save_pickle(tokens_sequences, '../pickle/tokens_sequences_no_spaces.pickle')
max_sentence_len = tokens_sequences['max_sentence_len']
sentences = tokens_sequences['sentences']
sentences.append([PLACEHOLDER])
print("MAX SENTENCE LENGHT: " + str(max_sentence_len))
print("NUMBER OF SENTENCIES: " + str(len(sentences)))

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
tokens_sequences = load_pickle('../pickle/tokens_sequences_no_spaces.pickle')
max_sentence_len = tokens_sequences['max_sentence_len']
tokens_to_exclude = TOKEN_TO_EXCLUDE

print("################################## GENERATOR ################################")

dataset = Dataset(word_model)
labels_path, img_paths = Dataset.load_paths_only(IMG_PATH)
labels_path_eval, img_paths_eval = Dataset.load_paths_only(IMG_PATH_EVAL)

build_generator = DataGenerator(img_paths, labels_path, word_model, max_code_len=max_sentence_len,
                                is_with_output_name=True, batch_size=1)
generator = iter_sequence_infinite(DataGenerator(img_paths, labels_path, word_model, max_code_len=max_sentence_len,
                                                 is_with_output_name=True, batch_size=32))
generator_val = iter_sequence_infinite(DataGenerator(img_paths_eval, labels_path_eval, word_model,
                                                     max_code_len=max_sentence_len, is_with_output_name=True,
                                                     batch_size=32))

print("################################## MODEL ################################")

# # new_model = Pix2codeW2VEmbedding(pretrained_weights=pretrained_weights)
model_instance = W2VCnnModel(w2v_pretrained_weights=pretrained_weights,
                        words=words,
                        image_count_words=voc.get_tokens(),
                        max_code_length=max_sentence_len,
                        dropout_ratio=0.1)
model_instance.compile()


# eval_set = '../datasets/web/train_features'
# labels_path, img_paths = Dataset.load_paths_only(eval_set)
# generator = DataGenerator(img_paths, labels_path, word_model, is_with_output_name=True, batch_size=1)
#
img_build, _ = build_generator.__getitem__(0)

# data = [(generator.__getitem__(i)) for i in range(len(img_paths))]
test = model_instance.predict(img_build)
# print(test)
# eval_code_error(new_model, data, img_paths, voc, max_sentence_len)

# code = model_instance.predict_image('../datasets/web/prove/0D1C8ADB-D9F0-48EC-B5AA-205BCF96094E.png', voc)
#
# print(code)

# new_model.load_weights('../instances/pix2code_original_w2v.h5')

print("################################## FIT ##################################")

training_steps = int(len(img_paths) / 32) * 5
val_steps = int(len(img_paths_eval) / 32)
model_save_path = '../instances/W2VCnnModel.h5'

# Do the transfer learning
transfer_learning_model_save_path = '../instances/CnnImageModel.h5'
model_instance.load_weights(transfer_learning_model_save_path, by_name=True)
# Do initial training with transferred layers locked
for layer_name in ['cnn_unit', 'counter_unit', 'ordering_1', 'ordering_2', 'ordering_3']:
    layer = model_instance.get_layer(layer_name)
    layer.trainable = False
loss = {word + "_count": 'mse' for word in model_instance.image_count_words}
loss.update({'code': 'sparse_categorical_crossentropy'})
loss_weights = {word + "_count": 1 / len(model_instance.image_count_words) for word in model_instance.image_count_words}
loss_weights.update({'code': 1.0})
model_instance.compile(loss=loss, loss_weights=loss_weights, optimizer=RMSprop(lr=0.0001, clipvalue=1.0))

early_stopping_patience = 5
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)

hist = model_instance.fit(generator, steps_per_epoch=training_steps,
                          validation_data=generator_val, validation_steps=val_steps,
                          epochs=1, callbacks=[checkpoint_cb, early_stopping_cb])
# Unlock and continue training
for layer_name in ['cnn_unit', 'counter_unit', 'ordering_1', 'ordering_2', 'ordering_3']:
    layer = model_instance.get_layer(layer_name)
    layer.trainable = True
loss = {word + "_count": 'mse' for word in model_instance.image_count_words}
loss.update({'code': 'sparse_categorical_crossentropy'})
loss_weights = {word + "_count": 1 / len(model_instance.image_count_words) for word in model_instance.image_count_words}
loss_weights.update({'code': 10.0})
model_instance.compile(loss=loss, loss_weights=loss_weights, optimizer=RMSprop(lr=0.00001, clipvalue=1.0))

early_stopping_patience = 10
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)

hist = model_instance.fit(generator, steps_per_epoch=training_steps,
                          validation_data=generator_val, validation_steps=val_steps,
                          epochs=1, callbacks=[checkpoint_cb, early_stopping_cb])

# new_model.fit([dataset.input_images, dataset.partial_sequences], dataset.next_words)
#
# checkpoint_filepath = '../instances/Pix2codeW2VEmbedding.h5'
#
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     save_best_only=True)
#
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     restore_best_weights=True,
#     patience=10)
#
# history = new_model.fit(generator,
#                         validation_data=generator_eval,
#                         epochs=4000,
#                         steps_per_epoch=int(len(img_paths) / 32) * 5,
#                         validation_steps=int(len(img_paths_eval) / 32),
#                         callbacks=[model_checkpoint_callback, early_stopping])

print("################################## PREDICT ##################################")

# image_to_predict = '/home/adamf42/Projects/pix2code/datasets/web/single/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.png'
#
# prediction = Pix2codeW2VEmbedding.predict_image(new_model, image_to_predict, word_model, max_sentence_len)
#
# print(prediction)

import gensim
from gensim.models import Word2Vec

from utils.costants import PLACEHOLDER
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

print("################################## GENERATOR ################################")

# dataset = Dataset(word_model)
labels_path, img_paths = Dataset.load_paths_only(IMG_PATH)
labels_path_eval, img_paths_eval = Dataset.load_paths_only(IMG_PATH_EVAL)
generator = DataGenerator(img_paths, labels_path, word_model, is_with_output_name=True)
generator_eval = DataGenerator(img_paths_eval, labels_path_eval, word_model, is_with_output_name=True)

print("################################## MODEL ################################")

# new_model = Pix2codeW2VEmbedding(pretrained_weights=pretrained_weights)
new_model = W2VCnnModel(pretrained_weights=pretrained_weights,
                        image_count_words=voc.get_tokens(),
                        dropout_ratio=0.1)
new_model.compile()
img_build, _ = generator.__getitem__(0)
# test = new_model.predict(img_build)
code = new_model.predict_image('../datasets/web/prove/0D1C8ADB-D9F0-48EC-B5AA-205BCF96094E.png', voc,
                               max_sentence_len)

print(code)

# new_model.load_weights('../instances/pix2code_original_w2v.h5')

print("################################## FIT ##################################")

# new_model.fit([dataset.input_images, dataset.partial_sequences], dataset.next_words)

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
#                         steps_per_epoch=int(len(img_paths) / 32) * 8,
#                         validation_steps=int(len(img_paths_eval) / 32),
#                         callbacks=[model_checkpoint_callback, early_stopping])

# print("################################## PREDICT ##################################")
#
# image_to_predict = '/home/adamf42/Projects/pix2code/datasets/web/single/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.png'
#
# prediction = Pix2codeW2VEmbedding.predict_image(new_model, image_to_predict, word_model, max_sentence_len)
#
# print(prediction)

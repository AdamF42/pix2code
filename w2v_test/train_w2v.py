import gensim
from gensim.models import Word2Vec
from tensorflow.python.keras.models import load_model

from w2v_test.dataset.dataset import Dataset
from utils.utils import get_token_sequences_with_max_seq_len
from w2v_test.generator.generator import DataGenerator
from w2v_test.models.pix2code_w2v_embedding import Pix2codeW2VEmbedding

IMG_W2V_TRAIN_DIR = '/home/adamf42/Projects/pix2code/datasets/web/all_data'
IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/training_set'

print("################################## GENSIM ##################################")

print('\nPreparing the sentences...')

tokens = get_token_sequences_with_max_seq_len(IMG_W2V_TRAIN_DIR)

max_sentence_len = tokens['max_sentence_len']
sentences = tokens['sentences']

print("MAX SENTENCE LENGHT: " + str(max_sentence_len))
print("NUMBER OF SENTENCIES: " + str(len(sentences)))

print('\nTraining word2vec...')
# word_model: Word2Vec = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=3, iter=200)
word_model: Word2Vec = gensim.models.Word2Vec.load('/home/adamf42/Projects/pix2code/w2v_test/word2vec.model')
pretrained_weights = word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)

print("emdedding_size: {}, vocab_size: {}".format(emdedding_size, vocab_size))

print("################################## DATASET ##################################")

# dataset = Dataset(word_model)
#
# dataset.load(IMG_PATH)
#
# dataset.create_word2vec_representation()
#
# print(dataset.partial_sequences.shape)

print("################################## MODEL ##################################")

new_model = Pix2codeW2VEmbedding(pretrained_weights=pretrained_weights)

new_model.compile()

labels, img_paths = Dataset.load_paths_only(IMG_PATH)

generator = DataGenerator(img_paths, labels, word_model)



# shape = [(None, 256, 256, 3), (None, 48)]
#
# new_model.build(shape)
#
# new_model.summary()

new_model: Pix2codeW2VEmbedding = load_model('/home/adamf42/Projects/pix2code/w2v_test/pix2code_new_w2v')


print("################################## FIT ##################################")

# new_model.fit([dataset.input_images, dataset.partial_sequences], dataset.next_words)

# new_model.fit(generator, epochs=10)

print("################################## PREDICT ##################################")

image_to_predict = '/home/adamf42/Projects/pix2code/datasets/web/single/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.png'

prediction = Pix2codeW2VEmbedding.predict_image(new_model, image_to_predict, word_model, max_sentence_len)

print(prediction)

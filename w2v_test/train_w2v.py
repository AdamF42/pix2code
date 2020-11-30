import os

import gensim
from gensim.models import Word2Vec

from w2v_test.dataset.dataset import Dataset
from w2v_test.generator.generator import DataGenerator
from w2v_test.models.pix2code_w2v_embedding import Pix2codeW2VEmbedding

IMG_W2V_TRAIN_DIR = '/home/adamf42/Projects/pix2code/datasets/web/all_data'
IMG_PATH = '/home/adamf42/Projects/pix2code/datasets/web/training_set'

print("################################## GENSIM ##################################")

print('\nPreparing the sentences...')

max_sentence_len = 0
sentences = []

for filename in os.listdir(IMG_W2V_TRAIN_DIR):
    if filename.endswith(".gui"):
        with open(os.path.join(IMG_W2V_TRAIN_DIR, filename)) as doc:
            document = []
            document.append("<START>")
            for line in doc.readlines():
                line = line.replace(" ", "  ") \
                    .replace(",", " ,") \
                    .replace("\n", " \n") \
                    .replace("{", " { ") \
                    .replace("}", " } ") \
                    .replace(",", " , ")
                tokens = line.split(" ")
                tokens = map(lambda x: " " if x == "" else x, tokens)
                tokens = filter(lambda x: False if x == " " else True, tokens)
                tokens = filter(lambda x: False if x == "\n" else True, tokens)
                for token in tokens:
                    document.append(token)
            document.append("<END>")
            if len(document) > max_sentence_len:
                max_sentence_len = len(document)
            sentences.append(document)

print("MAX SENTENCE LENGHT: " + str(max_sentence_len))
print("NUMBER OF SENTENCIES: " + str(len(sentences)))

print('\nTraining word2vec...')
word_model: Word2Vec = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=3, iter=200)
pretrained_weights = word_model.wv.vectors
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)

print("emdedding_size: {}, vocab_size: {}".format(emdedding_size, vocab_size))

print("################################## DATASET ##################################")

dataset = Dataset(word_model)

dataset.load(IMG_PATH)

dataset.create_word2vec_representation()

print(dataset.partial_sequences.shape)

print("################################## MODEL ##################################")

new_model = Pix2codeW2VEmbedding(pretrained_weights=pretrained_weights)

new_model.compile()

labels, img_paths = Dataset.load_paths_only(IMG_PATH)

generator = DataGenerator(img_paths, labels, word_model)

shape = [(None, 256, 256, 3), (None, 48)]

new_model.build(shape)

new_model.summary()

print("################################## FIT ##################################")

# new_model.fit([dataset.input_images, dataset.partial_sequences], dataset.next_words)

new_model.fit_generator(generator=generator)

print("################################## PREDICT ##################################")

image_to_predict = '/home/adamf42/Projects/pix2code/datasets/web/single/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.png'

prediction = Pix2codeW2VEmbedding.predict_image(new_model, image_to_predict, word_model, max_sentence_len)

print(prediction)

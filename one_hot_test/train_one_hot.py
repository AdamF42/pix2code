from one_hot_test.dataset.Dataset import Dataset
from w2v_test.dataset.utils import get_token_sequences_with_max_seq_len
from one_hot_test.generator.generator import DataGenerator
from one_hot_test.models.pix2code_one_hot_embedding import Pix2codeOneHotEmbedding


IMG_ONEHOT_TRAIN_DIR = '../datasets/web/eval_set'  # training_set

print("################################## DATASET ##################################")

print('\nPreparing the sentences...')

tokens = get_token_sequences_with_max_seq_len(IMG_ONEHOT_TRAIN_DIR)

max_sentence_len = tokens['max_sentence_len']
sentences = tokens['sentences']

print("MAX SENTENCE LENGHT: " + str(max_sentence_len))  # 117 (senza spazi), 278 (con gli spazi)
print("NUMBER OF SENTENCIES: " + str(len(sentences)))  # 1500 (diverse folder di input)

print('\nCreate one_hot encoding...')
dataset = Dataset()
for sentence in tokens['sentences']:
    for word in sentence:
        dataset.voc.append(word)
dataset.load_with_one_hot_encoding(IMG_ONEHOT_TRAIN_DIR)

print("voc: ", dataset.voc.vocabulary)
print("binary voc: ", dataset.voc.binary_vocabulary)
print("emb matrix: ", dataset.voc.embedding_matrix)

vocab_size = dataset.voc.size

print('Result embedding shape:', vocab_size)  # 18

print("################################## MODEL ##################################")

new_model = Pix2codeOneHotEmbedding(dataset.voc.embedding_matrix)

new_model.compile()

shape = [(None, 256, 256, 3), (None, 48)]
new_model.build(input_shape=shape)

labels, img_paths = Dataset.load_paths_only(IMG_ONEHOT_TRAIN_DIR)
print("len labels: ", len(labels))  # 250
print("len img_paths: ", len(img_paths))  # 250

generator = DataGenerator(img_paths, labels, dataset.voc)

print("################################## FIT ##################################")

new_model.fit(generator, epochs=10)

print("################################## PREDICT ##################################")

# image_to_predict = '../datasets/web/eval_set/0D99F46A-BEDB-444C-B948-246096DFEBD4.png'
#
# prediction = Pix2codeOneHotEmbedding.predict_image(new_model, image_to_predict, dataset.voc, max_sentence_len)
#
# print(prediction)

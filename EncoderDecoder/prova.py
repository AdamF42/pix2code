import time

import gensim
from gensim.models import Word2Vec

from EncoderDecoder.decoder import BahdanauAttention, Decoder
from EncoderDecoder.encoder import Encoder
from EncoderDecoder.generator import DataGenerator
from model.classes.Vocabulary import START_TOKEN
from utils.costants import PLACEHOLDER
from utils.dataset import Dataset
from w2v_test.models.VocabularyW2V import VocabularyW2V
import tensorflow as tf

IMG_PATH_TRAIN = '../datasets/web/validation_features'
BATCH_SIZE = 32

print('\nLoad word2vec...')
word_model: Word2Vec = gensim.models.Word2Vec.load('/home/adamf42/Projects/pix2code/w2v_test/word2vec.model')
pretrained_weights = word_model.wv.vectors
vocab_size, embedding_size = pretrained_weights.shape
voc = VocabularyW2V(word_model)

# encoder = Encoder(vocab_size=vocab_size, embedding_size=emdedding_size,
#                   enc_units=128, batch_sz=BATCH_SIZE,
#                   w2v_pretrained_weights=pretrained_weights)

encoder = Encoder(vocab_size=vocab_size, embedding_dim=embedding_size,
                  enc_units=128, batch_sz=BATCH_SIZE)

dataset = Dataset(word_model)
train_labels, train_paths = Dataset.load_paths_only(IMG_PATH_TRAIN)

test = list(voc.get_tokens())
test.remove(PLACEHOLDER)

generator = DataGenerator(train_paths, train_labels, word_model, output_names=test, max_code_len=117,
                          is_count_required=False, is_with_output_name=True, batch_size=BATCH_SIZE)

example_input_batch, _ = generator.__getitem__(0)
# example_input_batch = example_input_batch[0]

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


decoder = Decoder(vocab_size, embedding_size, 512, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  # mask = tf.math.logical_not(tf.math.equal(real, 0))
  # tf.print(real)
  # tf.print(mask)
  loss_ = loss_object(real, pred)

  # mask = tf.cast(mask, dtype=loss_.dtype)
  # loss_ *= mask
  tf.print(loss_)

  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([voc.word_to_index(START_TOKEN)] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    # for t in range(1, targ.shape[1]):
    # for t in range(1, len(targ)):

    # passing enc_output to the decoder
    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

    loss += loss_function(targ, predictions)


    # using teacher forcing
    dec_input = tf.expand_dims(targ, 1)

  batch_loss = (loss / int(len(targ)))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss



EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  # for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
  for step in range (0, generator.__len__()):

    inp, targ = generator.__getitem__(step)

    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    # if batch % 100 == 0:
    #   print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
    #                                                batch,
    #                                                batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  # if (epoch + 1) % 2 == 0:
  #   checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss ))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
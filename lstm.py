from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class LSTM_Config():
    PAD = 0
    EOS = 1

    vocab_size = 10
    input_embedding_size = 20

    encoder_hidden_units = 20
    decoder_hidden_units = encoder_hidden_units

class LSTM():

    def __init__(self, config):
        self.config = config

        self.set_variables()
        self.create_encoder()
        self.create_decoder()
        self.create_loss()

    def set_variables(self):
        self.encoder_inputs = tf.placeholder(shape=(None, None), 
                                             dtype=tf.int32, name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), 
                                              dtype=tf.int32, name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, None), 
                                             dtype=tf.int32, name='decoder_inputs')
        self.embeddings = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.input_embedding_size], -1.0, 1.0), 
                                 dtype=tf.float32)

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

    def create_encoder(self):
        # Build RNN cell
        self.encoder_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_units)

        _, self.encoder_final_state = tf.nn.dynamic_rnn(
            self.encoder_cell, self.encoder_inputs_embedded,
            dtype=tf.float32, time_major=True,
        )

    def create_decoder(self):
        self.decoder_cell = tf.contrib.rnn.LSTMCell(self.config.decoder_hidden_units)

        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                self.decoder_cell, self.decoder_inputs_embedded,
                initial_state=self.encoder_final_state,
                dtype=tf.float32, time_major=True, scope="plain_decoder")

        self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.config.vocab_size)

        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

    def create_loss(self):
        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.config.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits,
        )

        self.loss_op = tf.reduce_mean(self.stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)

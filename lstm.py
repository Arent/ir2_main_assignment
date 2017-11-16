from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class LSTM_Config():
    PAD = 0
    EOS = 1

    vocab_size = 10
    input_embedding_size = 20

    activation = tf.nn.relu

    context_units = 20
    question_units = context_units

class LSTM():

    def __init__(self, config):
        self.config = config

        self.set_variables()
        self.create_encoder()
        self.create_dense()
        self.create_loss()
        self.create_accuracy()

    def set_variables(self):
        # The encoder input consists of all the context sentences.
        self.context_input = tf.placeholder(shape=(None, None), 
                                             dtype=tf.int32, name='context_input')

        # The encoder input consists of all the context sentences.
        self.answer_labels = tf.placeholder(dtype=tf.int32)

        # The encoder input consists of all the context sentences.
        self.question_length = tf.placeholder(dtype=tf.int32)

        # We create embedings for each word in the vocab.
        self.embeddings = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.input_embedding_size]), 
                                 dtype=tf.float32)

        # The actual input will be the input converted to the created vocab embeddings.
        self.encoder_context_embedded = tf.nn.embedding_lookup(self.embeddings, self.context_input)
        # self.encoder_question_embedded = tf.nn.embedding_lookup(self.embeddings, self.question_input)

    def create_encoder(self):
        # Start with constructing a LSTM cell of the configured size.
        self.encoder_cell = tf.contrib.rnn.LSTMCell(self.config.context_units)

        # TODO explain this cell
        _, self.encoder_final_state = tf.nn.dynamic_rnn(
            self.encoder_cell, self.encoder_context_embedded,
            dtype=tf.float32, sequence_length=self.question_length, time_major=True,
        )

    def create_dense(self):
        embedded_question = tf.concat([self.encoder_final_state.c, self.encoder_final_state.h], -1)
        self.logits = tf.layers.dense(
                tf.layers.dense(embedded_question, self.config.question_units,
                                activation=None, use_bias=True),
                        self.config.vocab_size, activation=None, use_bias=True)


    def create_loss(self):
        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.answer_labels, depth=self.config.vocab_size, dtype=tf.float32),
            logits=self.logits,
        )

        self.loss_op = tf.reduce_mean(self.stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)

    def create_accuracy(self):
        self.prediction = tf.argmax(self.logits, axis=-1, output_type=self.answer_labels.dtype)

        self.accuracy_op = tf.reduce_mean(tf.cast(tf.equal(self.answer_labels, self.prediction), tf.float32))
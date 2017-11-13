import tensorflow as tf
import numpy as np

class QALSTM:

  def __init__(self, vocab_size, questions, answers, question_lengths, embedding_size=64, num_units=64, activation=tf.nn.relu, encoder_type="uni"):
    self.embedding_size = embedding_size
    self.num_units = num_units
    self.activation = activation
    self.vocab_size = vocab_size
    self.encoder_type = encoder_type

    self.question = questions
    self.answer = answers
    self.question_length = question_lengths

    self._create_model()

  def _create_model(self):

    # retrieve embeddings
    embedding_matrix = tf.get_variable("embedding_matrix", \
        [self.vocab_size, self.embedding_size], dtype=tf.float32)
    embeddings = tf.nn.embedding_lookup(embedding_matrix, self.question) # [1, time, emb_size]

    # zero_state_c, zero_state_h = cell.zero_state(1, dtype=tf.float32)

    # Run the LSTM.
    if self.encoder_type == "uni":
      cell = tf.contrib.rnn.LSTMCell(self.num_units,
          activation=self.activation)
      outputs, final_state = tf.nn.dynamic_rnn(cell, embeddings,
          dtype=tf.float32, sequence_length=self.question_length)
      final_state_c = final_state.c
      final_state_h = final_state.h
    else:
      fw_cell = tf.contrib.rnn.LSTMCell(self.num_units,
          activation=self.activation)
      bw_cell = tf.contrib.rnn.LSTMCell(self.num_units,
          activation=self.activation)

      # Run the bidirectional RNN.
      bi_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
          fw_cell,
          bw_cell,
          embeddings,
          sequence_length=self.question_length,
          dtype=tf.float32)

      # Concatenate the forward and backward layer to create the encoder output.
      outputs = tf.concat(bi_outputs, axis=-1)
      final_state_c = tf.concat([final_state[0].c, final_state[1].c], axis=-1)
      final_state_h = tf.concat([final_state[1].h, final_state[1].h], axis=-1)

    # Predict the answer word.
    embedded_question = tf.concat([final_state_c, final_state_h], -1)
    self.logits = tf.layers.dense(
        tf.layers.dense(embedded_question, self.num_units,
                        activation=self.activation, use_bias=True),
        self.vocab_size, activation=None, use_bias=True)

  # def __neural_net(self, num_layers, num_units, output_size, use_bias=True, name="neural_net"):

  def loss(self):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.answer,
        logits = self.logits,
        name="cross_entropy_loss")
    return tf.reduce_mean(cross_entropy)

  def accuracy(self):
    prediction = tf.argmax(self.logits, axis=-1, output_type=self.answer.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(self.answer, prediction), tf.float32))

import tensorflow as tf
import numpy as np

class QARNN:

  def __init__(self, mode, vocab_size, embedding_size=64, num_units=64, encoder_type="bi", keep_prob=0.7,
      cell_type=tf.contrib.rnn.BasicLSTMCell, num_output_hidden=[256], num_enc_layers=1, merge_mode="concat",
      optimizer=tf.train.AdamOptimizer, learning_rate=0.001, max_gradient_norm=1.0):
    self.embedding_size = embedding_size
    self.num_units = num_units
    self.vocab_size = vocab_size
    self.encoder_type = encoder_type
    self.keep_prob = keep_prob
    self.cell_type = cell_type
    self.num_output_hidden = num_output_hidden
    self.num_enc_layers = num_enc_layers
    self.merge_mode = merge_mode
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.max_gradient_norm = max_gradient_norm
    self.mode = mode

  # Creates an encoder on the given sequence, returns the final state of the encoder.
  def create_encoder(self, name, sequence, sequence_length):

    with tf.variable_scope(name) as scope:
      embedding_matrix = tf.get_variable("embedding_matrix", \
          [self.vocab_size, self.embedding_size], dtype=tf.float32)
      embeddings = tf.nn.embedding_lookup(embedding_matrix,
          sequence) # [batch, time, emb_size]

      if self.encoder_type == "uni":

        # Build a multilayer RNN encoder.
        cells = []
        for _ in range(self.num_enc_layers):
          cell = self.cell_type(self.num_units)

          # Perform dropout during training.
          if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)

          cells.append(cell)

        cell = tf.contrib.rnn.MultiRNNCell(cells)

        _, final_state = tf.nn.dynamic_rnn(cell, embeddings,
            dtype=tf.float32, sequence_length=sequence_length)

        # For multilayer RNNs, use the final layer's state.
        final_state = final_state[-1]

        # For tuple states such as LSTM, return only h as the encoded
        if isinstance(final_state, tuple):
          final_state = final_state.h

        return final_state
      else:

        # Bidirectional encoders can have only 1 layer in our implementation.
        assert self.num_enc_layers == 1

        fw_cell = self.cell_type(self.num_units)
        bw_cell = self.cell_type(self.num_units)

        # Perform dropout during training.
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
          fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
          bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)

        # Run the bidirectional RNN.
        _, bi_final_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            embeddings,
            sequence_length=sequence_length,
            dtype=tf.float32)

        # For tuple states such as LSTM, return only h as the encoded, concatenate
        # forward and backward RNN outputs.
        if isinstance(final_state, tuple):
          final_state = tf.concat([bi_final_state[1].h, bi_final_state[1].h], axis=-1)
        else:
          final_state = tf.concat(bi_final_state, axis=-1)

    return final_state

  # Combine the context and question encoding.
  def merge_encodings(self, encoded_context, encoded_question):
    if self.merge_mode == "sum":
      merged_encoding = encoded_context + encoded_question
    elif self.merge_mode == "concat":
      merged_encoding = tf.concat([encoded_context, encoded_question], axis=-1)
    else:
      print("ERROR: unknown merge mode")
      merged_encoding = None
    return merged_encoding

  # Create an MLP decoder.
  def create_decoder(self, decoder_inputs):
    with tf.variable_scope("decoder"):

      # Create the hidden layers.
      prev_h = decoder_inputs
      for num_hidden_units in self.num_output_hidden:
        h = tf.layers.dense(prev_h, num_hidden_units,
            activation=tf.nn.relu, use_bias=True)

        # Perform dropout when training.
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
          h = tf.nn.dropout(h, self.keep_prob)

        prev_h = h

      # Predict the answer word using a dense output layer to vocab size.
      logits = tf.layers.dense(h, self.vocab_size,
          activation=None, use_bias=False)

    return logits

  def loss(self, logits, answer):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=answer,
        logits=logits,
        name="cross_entropy_loss")
    return tf.reduce_mean(cross_entropy)

  def train_step(self, loss):
    optimizer = self.optimizer(self.learning_rate)

    # Clip the gradients before applying them.
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients,
        self.max_gradient_norm)

    # Apply the gradients
    train_op = optimizer.apply_gradients(
        zip(clipped_gradients, params))

    return train_op

  def accuracy(self, logits, answer):
    prediction = tf.argmax(logits, axis=-1, output_type=answer.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(answer, prediction), tf.float32))

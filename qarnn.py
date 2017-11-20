import tensorflow as tf
import numpy as np

class QARNN:

  def __init__(self, mode, vocab_size, contexts, questions, answers, context_lengths, question_lengths,
      embedding_size=64, num_units=64, activation=tf.nn.relu, encoder_type="bi", keep_prob=0.7,
      cell_type="gru", num_output_hidden=[256]):
    self.embedding_size = embedding_size
    self.num_units = num_units
    self.activation = activation
    self.vocab_size = vocab_size
    self.encoder_type = encoder_type
    self.keep_prob = keep_prob
    self.cell_type = cell_type
    self.num_output_hidden = num_output_hidden
    self.mode = mode

    self.context = contexts
    self.question = questions
    self.answer = answers
    self.context_length = context_lengths
    self.question_length = question_lengths

    self._create_model()

  def _create_model(self):

    # mode = "c_plus_q" # c_plus_q|shared|separate
    # emb_mode = "shared" # shared|separate
    # c + q
    # c_and_q = tf.concat([self.context, self.question], axis=-1)
    # joint_embeddings = tf.nn.embedding_lookup(embedding_matrix, c_and_q)

    # retrieve embeddings
    context_embedding_matrix = tf.get_variable("context_embedding_matrix", \
        [self.vocab_size, self.embedding_size], dtype=tf.float32)
    question_embedding_matrix = tf.get_variable("question_embedding_matrix", \
        [self.vocab_size, self.embedding_size], dtype=tf.float32)
    context_embeddings = tf.nn.embedding_lookup(context_embedding_matrix,
        self.context)
    question_embeddings = tf.nn.embedding_lookup(question_embedding_matrix,
        self.question) # [1, time, emb_size]

    # Perform dropout on the embeddings. (Is this common?)
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      context_embeddings = tf.nn.dropout(context_embeddings, self.keep_prob)
      question_embeddings = tf.nn.dropout(question_embeddings, self.keep_prob)

    # Run the LSTM on the context.
    outputs, final_state = self._create_encoder(context_embeddings, self.context_length,
        name="context_encoder", reuse=False)
    encoded_context = final_state

    # Run the LSTM on the question.
    outputs, final_state = self._create_encoder(question_embeddings, self.question_length,
        name="question_encoder", reuse=False)
    encoded_question = final_state

    # Combine the story and question.
    encoder_output = tf.concat([encoded_context, encoded_question], axis=-1)
    # encoder_output = encoded_context + encoded_question

    # Perform dropout when training.
    prev_h = encoder_output
    for num_hidden_units in self.num_output_hidden:
      h = tf.layers.dense(prev_h, num_hidden_units,
          activation=self.activation, use_bias=False)
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        h = tf.nn.dropout(h, self.keep_prob)
      prev_h = h

    # Predict the answer word.
    self.logits = tf.layers.dense(h, self.vocab_size,
        activation=None, use_bias=False)

  def _create_cell(self):
    if self.cell_type == "gru":
      cell = tf.contrib.rnn.GRUCell(self.num_units,
          activation=self.activation)
    elif self.cell_type == "lstm":
      cell = tf.contrib.rnn.LSTMCell(self.num_units,
          activation=self.activation)
    else:
      print("ERROR: unknown cell type")
      cell = None
    return cell

  def _create_encoder(self, inputs, sequence_length, name="encoder", reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
      if self.encoder_type == "uni":
        cell = self._create_cell()
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,
            dtype=tf.float32, sequence_length=sequence_length)

        if self.cell_type == "lstm":
          final_state = final_state.h
      else:
        fw_cell = self._create_cell()
        bw_cell = self._create_cell()

        # Run the bidirectional RNN.
        bi_outputs, bi_final_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            sequence_length=sequence_length,
            dtype=tf.float32)

        # Concatenate the forward and backward layer to create the encoder output.
        outputs = tf.concat(bi_outputs, axis=-1)

        if self.cell_type == "lstm":
          final_state = tf.concat([bi_final_state[1].h, bi_final_state[1].h], axis=-1)
        else:
          final_state = tf.concat(bi_final_state, axis=-1)
    return outputs, final_state

  def loss(self):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.answer,
        logits = self.logits,
        name="cross_entropy_loss")
    return tf.reduce_mean(cross_entropy)

  def accuracy(self):
    prediction = tf.argmax(self.logits, axis=-1, output_type=self.answer.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(self.answer, prediction), tf.float32))

import tensorflow as tf
import numpy as np

class QARNN:

  def __init__(self, mode, vocab, vocab_size, embedding_size=64, num_units=64, encoder_type="uni", keep_prob=0.7,
      cell_type=tf.contrib.rnn.LSTMCell, num_output_hidden=[256], num_enc_layers=1, merge_mode="concat",
      optimizer=tf.train.AdamOptimizer, learning_rate=0.001, max_gradient_norm=1.0, max_infer_length=10, attention=False):
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
    self.max_infer_length = max_infer_length
    self.vocab = vocab
    self.attention = attention

  def create_embeddings(self, sequence, name=None, embedding_matrix=None):
    if embedding_matrix is None:
      embedding_matrix = tf.get_variable(name, [self.vocab_size, self.embedding_size],
          dtype=tf.float32)

    return tf.nn.embedding_lookup(embedding_matrix,
        sequence), embedding_matrix # [batch, time, emb_size]

  # Creates an encoder on the given sequence, returns the final state of the encoder.
  def create_encoder(self, name, embeddings, sequence_length):

    with tf.variable_scope(name) as scope:

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

        encoder_outputs, final_state = tf.nn.dynamic_rnn(cell, embeddings,
            dtype=tf.float32, sequence_length=sequence_length)

        # For multilayer RNNs, use the final layer's state.
        final_state = final_state[-1]
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
        bi_outputs, bi_final_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            embeddings,
            sequence_length=sequence_length,
            dtype=tf.float32)

        # Concatenate the forward and backward layer to create the encoder output.
        encoder_outputs = tf.concat(bi_outputs, axis=-1)

        # For tuple states such as LSTM, return only h as the encoded, concatenate
        # forward and backward RNN outputs.
        if isinstance(bi_final_state[0], tf.contrib.rnn.LSTMStateTuple):
          c = tf.concat([bi_final_state[0].c, bi_final_state[1].c], axis=-1)
          h = tf.concat([bi_final_state[0].h, bi_final_state[1].h], axis=-1)
          final_state = tf.contrib.rnn.LSTMStateTuple(c, h)
        else:
          final_state = tf.concat(bi_final_state, axis=-1)

    return encoder_outputs, final_state

  # Combine the context and question states.
  def merge_states(self, final_context_state, final_question_state):
    if self.merge_mode == "sum":
      if isinstance(final_context_state, tf.contrib.rnn.LSTMStateTuple):
        c = final_context_state.c + final_question_state.c
        h = final_context_state.h + final_question_state.h
        merged_state = tf.contrib.rnn.LSTMStateTuple(c, h)
      else:
        merged_state = final_context_state + final_question_state
    elif self.merge_mode == "concat":
      if isinstance(final_context_state, tf.contrib.rnn.LSTMStateTuple):
        c = tf.concat([final_context_state.c, final_question_state.c], axis=-1)
        h = tf.concat([final_context_state.h, final_question_state.h], axis=-1)
        merged_state = tf.contrib.rnn.LSTMStateTuple(c, h)
      else:
        merged_state = tf.concat([final_context_state, final_question_state], axis=-1)
    else:
      print("ERROR: unknown merge mode")
      merged_state = None
    return merged_state

  def create_rnn_decoder(self, decoder_emb_inputs, input_length,
      initial_state, embedding_matrix, attention_states=None):
    sos_id = tf.cast(self.vocab.lookup(tf.constant("<s>")), tf.int32)
    eos_id = tf.cast(self.vocab.lookup(tf.constant("</s>")), tf.int32)
    with tf.variable_scope("decoder") as decoder_scope:

      num_units = self.num_units if self.encoder_type == "uni" else 2 * self.num_units
      num_units = 2 * num_units if self.merge_mode == "concat" and not self.attention else num_units
      cell = self.cell_type(num_units)

      if self.attention:
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
            attention_states)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell,
            attention_mechanism)
        batch_size = tf.shape(decoder_emb_inputs)[0]
        initial_state = cell.zero_state(batch_size, initial_state.dtype).clone(cell_state=initial_state)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs,
            input_length)
        maximum_iterations = tf.shape(decoder_emb_inputs)[1]
      else:
        batch_size = tf.shape(decoder_emb_inputs)[0]
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_matrix,
            tf.fill([batch_size], sos_id), eos_id)
        maximum_iterations = 10

      projection_layer = tf.layers.Dense(self.vocab_size,
          use_bias=False, name="output_projection")

      decoder = tf.contrib.seq2seq.BasicDecoder(
          cell, helper, initial_state,
          output_layer=projection_layer)

      outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
          impute_finished=True, scope=decoder_scope, maximum_iterations=maximum_iterations)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        logits = outputs.rnn_output
        return logits
      else:
        return outputs.sample_id

  def loss(self, logits, answer, answer_length):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=answer,
        logits=logits,
        name="cross_entropy_loss")

    # Mask padded batch elements.
    batch_size = tf.shape(logits)[0]
    max_time = tf.shape(logits)[1]
    mask = tf.sequence_mask(answer_length, max_time,
        dtype=logits.dtype)

    loss = tf.reduce_sum(cross_entropy * mask) / tf.to_float(batch_size)
    return loss

  def train_step(self, loss):
    optimizer = self.optimizer(self.learning_rate) #, beta1=0.) # TODO adam specific

    # Clip the gradients before applying them.
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients,
        self.max_gradient_norm)

    # Apply the gradients
    train_op = optimizer.apply_gradients(
        zip(clipped_gradients, params))

    return train_op

  def accuracy(self, logits, answer, answer_length):
    prediction = tf.argmax(logits, axis=-1, output_type=answer.dtype)

    # Don't include the end-of-sentence tokens in accuracy calculations.
    answer_length = answer_length - 1

    # Mask padded batch elements.
    max_time = tf.shape(answer)[1]
    mask = tf.sequence_mask(answer_length, max_time,
        dtype=logits.dtype)

    return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(answer, prediction), tf.float32) * mask,
        reduction_indices=[1]) / tf.cast(answer_length, tf.float32))

  def perplexity(self, loss, batch_size, answer_length):
    return tf.exp((loss * tf.cast(batch_size, tf.float32)) / tf.cast(tf.reduce_sum(answer_length), tf.float32))

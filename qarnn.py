import tensorflow as tf
import numpy as np


def softmax_with_sequence_length(logits, sequence_lengths):
    '''Return the softmax for each row until the sequence_length'''
    #Create boolean array from sequence_lengths
    #eg: sequence_lengths = [1,4,5], max_length = 5 (and batch_size is 3)
    #The boolean array(with shape: batch * max_time)  will be:
    # 1, 0, 0, 0, 0
    # 1, 1, 1, 1, 0
    # 1, 1, 1, 1 ,1
    batch_size = tf.shape(logits)[0]
    max_length = tf.shape(logits)[1]
    indices = tf.reshape(tf.tile(tf.range(max_length),[batch_size]), [batch_size,max_length])
    boolean_array = tf.cast(indices < sequence_lengths, tf.float32)    
    
    # Calculate regular softmax. but multiply tf.exp(x) with the boolean_array
    logits = tf.exp(logits -tf.reduce_max(logits,axis=1)[:,tf.newaxis]) * boolean_array
    return logits / tf.reduce_sum(logits,axis=1)[:,tf.newaxis]


class QARNN:

  def __init__(self, mode, vocab_size, embedding_size=64, num_units=64, encoder_type="bi", keep_prob=0.7,
      cell_type=tf.contrib.rnn.BasicLSTMCell, num_output_hidden=[256], num_enc_layers=1, merge_mode="concat",
      optimizer=tf.train.AdamOptimizer, learning_rate=0.001, max_gradient_norm=1.0, use_attention=False):
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
    self.use_attention = use_attention

    if self.num_enc_layers > 1 or self.encoder_type == 'bi':
      #Attention only implemented for simple Recurrent nets
      if self.use_attention:
        raise NotImplementedError

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

        output, final_state = tf.nn.dynamic_rnn(cell, embeddings,
            dtype=tf.float32, sequence_length=sequence_length)

        # For multilayer RNNs, use the final layer's state.
        final_state = final_state[-1]

        # For tuple states such as LSTM, return only h as the encoded
        if isinstance(final_state, tuple):
          final_state = final_state.h

        #For attention we need need the output of the context and the
        #Final embedded layer of the question
        if self.use_attention:
          return output, final_state
        else:
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
      h = prev_h
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


  def create_attention_layer(self, encoded_output_vector, output_matrix, output_lengths):
    '''This function returns the attenttion of the encoded output vetor 
    based on the columnd of the output_matrx 
    output_matrix: [batch_size, max_time, num_units]
    outout_lengths: [batch_size]
    encoded_output_vector: [batch_size, num_units]
    '''
    output_lengths = output_lengths[:,tf.newaxis]
    max_time = tf.shape(output_matrix)[1]
    batch_size = tf.shape(output_matrix)[0]


    with tf.variable_scope('attention'): 
      #First calculate the attention weights using a weighted dot product similarity.
      with tf.variable_scope('similarity_weights'): #, 
        # similarity_weights = tf.get_variable('Ws', shape=[self.num_units, self.num_units], dtype=tf.float32)[tf.newaxis,:]

        init_ws=tf.random_uniform(shape=[self.num_units, self.num_units], minval=-0.02,maxval=0.02) + tf.eye(self.num_units)
        similarity_weights = tf.get_variable('Ws', initializer=init_ws, dtype=tf.float32)[tf.newaxis,:]

        similarity_weights = tf.tile(similarity_weights, [batch_size,1,1])

      with tf.variable_scope('attention_weights'):
        encoded_output_vector = encoded_output_vector[:,:,tf.newaxis]
        
        attention_weights_input = tf.matmul(tf.matmul(output_matrix, similarity_weights), encoded_output_vector) 
        attention_weights = softmax_with_sequence_length(logits=tf.squeeze(attention_weights_input), sequence_lengths=output_lengths)
        attention_weights = attention_weights[:,:,tf.newaxis]
        

      with tf.variable_scope('context_vector'):
        #The context vector is the weighted sum over the outputs, using the learned attention weights
        output_matrix_bht = tf.transpose(output_matrix, [0,2,1])
        context_vector =tf.squeeze(tf.matmul(output_matrix_bht, attention_weights))

      #the final attention vector is learned using the concatination of the context vector
      #and the encoded output vector with an extra non_linearity
      with tf.variable_scope('attention_vector'):
        encoded_output_vector = tf.squeeze(encoded_output_vector)
        merged_context_output = tf.concat([context_vector, encoded_output_vector], axis=1)
        attention_weight_matrix = tf.get_variable('Wa', shape=[self.num_units*2, self.num_units]) 
        attention_vector = tf.tanh(tf.matmul(merged_context_output, attention_weight_matrix))

    return attention_vector


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

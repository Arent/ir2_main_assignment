import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ABCNN:
  def __init__(self, mode, vocab_size, embedding_size=64, encoder_type="bi", keep_prob=0.7, merge_mode="concat",
      optimizer=tf.train.AdamOptimizer, learning_rate=0.001):
    self.embedding_size = embedding_size
    self.vocab_size = vocab_size
    self.encoder_type = encoder_type
    self.keep_prob = keep_prob
    self.merge_mode = merge_mode
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.mode = mode
    self.h_size = 64 # Todo make this a parameter
    self.h_size2 = 32 # Todo make this a parameter

  # Todo add regularization ?
  def _get_weights_variable(self, name, shape):
    return tf.get_variable(name,
                           shape=shape,
                           initializer=tf.random_normal_initializer(stddev=0.1))
  @staticmethod
  def _get_bias_variable(name, shape):
    return tf.zeros(shape=shape, name=name)

  @staticmethod
  def _conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

  def create_encoder(self, name, sequence, k):
    with tf.variable_scope(name):
      embedding_matrix = tf.get_variable("embedding_matrix", \
                                         [self.vocab_size, self.embedding_size], dtype=tf.float32)

      embeddings = tf.nn.embedding_lookup(embedding_matrix,
                                            sequence)  # [batch, time, emb_size]

      # Todo rewrite this as a loop..
      # Todo: add dropout
      # Todo: add batch_normalization
      # Todo: add GLU
      # Todo: add Residual connections
      # Todo: return original input vector

      with tf.name_scope('conv1'):
        W_conv1 = self._get_weights_variable("W1", [k, self.embedding_size, self.h_size])
        b_conv1 = self._get_bias_variable("B1", [self.h_size ])
        h_conv1 = tf.nn.relu(self._conv1d(embeddings, W_conv1) + b_conv1)
        tf.summary.histogram("W1", W_conv1)
      with tf.name_scope('conv2'):
        W_conv2 = self._get_weights_variable("W2", [k, self.h_size , self.h_size ])
        b_conv2 = self._get_bias_variable("B2", [self.h_size ])
        h_conv2 = tf.nn.relu(self._conv1d(h_conv1, W_conv2) + b_conv2)
        tf.summary.histogram("W2", W_conv2)
      with tf.name_scope('conv3'):
        W_conv3 = self._get_weights_variable("W3", [k, self.h_size , self.h_size2])
        b_conv3 = self._get_bias_variable("B3", [self.h_size2])
        h_conv3 = tf.nn.relu(self._conv1d(h_conv2, W_conv3) + b_conv3)
        tf.summary.histogram("W3", W_conv3)
    return h_conv3

  def train_step(self, loss):
    return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

  @staticmethod
  def create_attention_matrix(context, question):
    mat = tf.einsum('aij,akj->aik',context, question)
    return mat

    # Create an MLP decoder.
  def create_decoder(self, attention_matrix, max_c_length, max_q_length):
    with tf.variable_scope("decoder"):
      # flatten attention matrix
      # attention_vector = tf.contrib.layers.flatten(attention_matrix)
      attention_vector = tf.reshape(attention_matrix, [-1, max_c_length * max_q_length], name="flat_attention")

      # Create MLP based on the attention matrix.
      with tf.name_scope('fc1'):
        W_fc1 = self._get_weights_variable("w_f1", [max_c_length * max_q_length, self.h_size])
        b_fc1 = self._get_bias_variable("b_f1", [self.h_size])
        h_fc1 = tf.nn.relu(tf.matmul(attention_vector, W_fc1) + b_fc1, name="relu_f1")
      with tf.name_scope('fc2'):
        W_fc2 = self._get_weights_variable("w_f2", [self.h_size, self.vocab_size])
        b_fc2 = self._get_bias_variable("b_f2", [self.vocab_size])
        logits = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2, name="relu_f2")

    return logits

  def loss(self, logits, answer):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=answer,
        logits=logits,
        name="cross_entropy_loss")
    return tf.reduce_mean(cross_entropy)

  def train_step(self, loss):
    train_op = self.optimizer(self.learning_rate).minimize(loss)

    return train_op

  def accuracy(self, logits, answer):
    prediction = tf.argmax(logits, axis=-1, output_type=answer.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(answer, prediction), tf.float32))

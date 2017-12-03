import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class ABCNN:
    def __init__( self, vocab_size, embedding_size=32, keep_prob=0.7,
                  optimizer=tf.train.AdamOptimizer, learning_rate=0.001, num_layers=2 ):
        self.vocab_size = vocab_size
        self.keep_prob = keep_prob
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_regularizer = tf.contrib.layers.l2_regularizer(0.01)
        self.num_layers = num_layers

        # Set h_size equal to embedding size
        self.h_size = embedding_size

    def _get_weights_variable( self, name, shape ):
        return tf.get_variable(name=name,
                               shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=self.weight_regularizer)

    def _get_bias_variable( self, name, shape ):
        return tf.get_variable(name=name,
                               shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=self.weight_regularizer)

    @staticmethod
    def _conv1d( x, W ):
        return tf.nn.conv1d(x, W, stride=1, padding='SAME')

    def create_embedding( self, name, seq, seq_len ):
        with tf.variable_scope(name):
            w_embedding = tf.get_variable("word_embedding", [self.vocab_size, self.h_size], dtype=tf.float32)
            p_embedding = tf.get_variable("positional_embedding", [seq_len, self.h_size], dtype=tf.float32)

            w = tf.nn.embedding_lookup(w_embedding, seq)  # [batch, time, emb_size]
            p = tf.nn.embedding_lookup(p_embedding, tf.range(seq_len))  # [batch, time, emb_size]
            e = w + p
            return e

    def create_encoder( self, name, e, k ):
        with tf.variable_scope(name):
            # Todo: add batch_normalization

            # Initialize the conv_output of the loop with the initial embedding(word + pos) of the input
            conv_output = e
            for i in range(1, self.num_layers + 1):
                with tf.name_scope('conv' + str(i)):
                    W_conv = self._get_weights_variable("W" + str(i), [k, self.h_size, self.h_size])
                    b_conv = self._get_bias_variable("B" + str(i), [self.h_size])

                    # weights for the GLU
                    W_conv2 = self._get_weights_variable("W" + str(i) + "2", [k, self.h_size, self.h_size])
                    b_conv2 = self._get_bias_variable("B" + str(i) + "2", [self.h_size])

                    # GLU operation
                    A1 = self._conv1d(conv_output, W_conv) + b_conv
                    B1 = self._conv1d(conv_output, W_conv2) + b_conv2
                    GLU = A1 * tf.nn.sigmoid(B1)

                    # Add residual connection
                    res_con = GLU + conv_output
                    conv_output = tf.nn.dropout(res_con, self.keep_prob)

        return conv_output

    @staticmethod
    def create_attention_matrix( context, question ):
        # Dot product of the context and question, result = [batch_size x context_max_len x question_max_len]
        mat = tf.einsum('ijl,ikl->ijk', context, question)

        # Take the average over the question dimension
        mat = tf.reduce_mean(mat, -1)

        attention_matrix = tf.nn.softmax(mat, 1)  # [batch_size x context_max_len]
        return attention_matrix

    def create_decoder( self, attention_matrix, context_output, context, max_c_length, max_q_length ):
        with tf.variable_scope("decoder"):
            # Multiply attention with context and sum over the len of the context
            attention_vector = tf.einsum('ijk,ij->ik', context_output + context, attention_matrix)  # [batch_size x h_size]

            with tf.name_scope('fc'):
                # No bias to avoid giving the same answer to every question.
                W_fc = self._get_weights_variable("w_f1", [self.h_size, self.vocab_size])
                logits = tf.nn.relu(tf.matmul(attention_vector, W_fc), name="relu_f")
        return logits

    def loss( self, logits, answer ):
        with tf.name_scope("loss"):
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=answer,
                logits=logits,
                name="cross_entropy_loss"))

            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(self.weight_regularizer, reg_variables)
            tf.summary.scalar('regularization loss', reg_term)

            loss = cross_entropy + reg_term
            tf.summary.scalar('full loss', loss)

        return loss

    def train_step( self, loss ):
        with tf.name_scope("train_step"):
            train_op = self.optimizer(self.learning_rate).minimize(loss)

        return train_op

    def accuracy( self, logits, answer ):
        with tf.name_scope("accuracy"):
            prediction = tf.argmax(logits, axis=-1, output_type=answer.dtype)
            acc = tf.reduce_mean(tf.cast(tf.equal(answer, prediction), tf.float32))
        return acc
import tensorflow as tf
import numpy as np
import argparse
import os
import random
from nltk.tokenize.moses import MosesTokenizer

import utils

# Parse cmd line args.
parser = argparse.ArgumentParser(description='Deep Learning '
                    'to Rank Assignment')
parser.add_argument('--data_dir', type=str, default=None,
                    help='Directory to store/load model.')
parser.add_argument('--vocab', type=str, default=None,
                    help='Vocabulary file')
parser.add_argument('--task', type=str, default='1',
                    help='Task number')
parser.add_argument('--batch_size', type=int, default=80,
                    help='Batch size')
parser.add_argument('--seperate_context', type=bool, default=True,
                    help='Seprate context from questions')

args = parser.parse_args()

# Load the vocabulary.
word2id, id2word = utils.load_vocab(args.vocab)
vocab_size = len(word2id)
tokenizer = MosesTokenizer()

embedding_size = 25
hidden_state_size_context = 45
hidden_state_size_question = 45
batch_size = 1
learning_rate = 1e-3
epochs = 10

# Load the training / testing data.
train, test = utils.load_data(args.task, args.task, args.data_dir, word2id, tokenizer, args.batch_size, args.seperate_context)
train_examples_num = len(train)
train_context, train_question, train_answer  = list(zip(*train))
test_context, test_question, test_answer  = list(zip(*train))


max_length = max(len(l) for l in train_context + test_context +
        train_question + test_question + train_answer + test_answer)


train_context_padded, train_context_lengts = utils.pad_results(train_context, max_length)
train_question_padded, train_question_lengts = utils.pad_results(train_question, max_length)
test_context_padded, test_context_lengts = utils.pad_results(test_context, max_length)
test_question_padded, test_question_lengts = utils.pad_results(test_question, max_length)




lengths_question = tf.placeholder(dtype=tf.int32, shape=[batch_size])
lengths_context = tf.placeholder(dtype=tf.int32, shape=[batch_size])
answer = tf.placeholder(dtype=tf.int32, shape=[batch_size])
rnn_input_context = tf.placeholder(tf.int32, shape=[batch_size, max_length])
rnn_input_question = tf.placeholder(tf.int32, shape=[batch_size, max_length])



embedding_matrix_context = tf.get_variable("embedding_matrix_context", \
        [vocab_size, embedding_size], dtype=tf.float32)
embeddings_context = tf.nn.embedding_lookup(embedding_matrix_context, rnn_input_context) # [1, time, emb_size]

embedding_matrix_question = tf.get_variable("embedding_matrix_question", \
        [vocab_size, embedding_size], dtype=tf.float32)
embeddings_question = tf.nn.embedding_lookup(embedding_matrix_question, rnn_input_question) # [1, time, emb_size]


# Build RNN cell
with tf.variable_scope("context"):
    lstm_context = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_size_context)
    # initial_state_context = lstm_context.zero_state(batch_size, tf.float32)
    encoder_outputs_context, encoder_state_context = tf.nn.dynamic_rnn(
        cell=lstm_context, inputs=embeddings_context, dtype=tf.float32,
        sequence_length=lengths_context, time_major=False)#, initial_state =initial_state_context)

with tf.variable_scope("question"):
    lstm_question = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_size_question)
    # initial_state_question = lstm_context.zero_state(batch_size, tf.float32)
    encoder_outputs_question, encoder_state_question = tf.nn.dynamic_rnn(
        lstm_question, embeddings_question, dtype=tf.float32,
        sequence_length=lengths_question, time_major=False)#, initial_state=initial_state_question)





combined_context_question = encoder_state_context[1] + encoder_state_question[1]
prediction_weights = tf.get_variable("prediction_weights", [hidden_state_size_context, vocab_size])
logits = tf.matmul(combined_context_question, prediction_weights)

labels = labels=tf.one_hot(indices=answer, depth=vocab_size)
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)


optimizer = tf.train.AdamOptimizer(learning_rate)

gradients, variables = zip(*optimizer.compute_gradients(cross_entropy_loss))
gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
train_op = optimizer.apply_gradients(zip(gradients, variables))


# train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for i in range(train_examples_num):
            answer_data = np.array(train_answer[i])
            o, train_loss = sess.run([train_op, cross_entropy_loss] ,
                    feed_dict={lengths_question:train_question_lengts[:,i] , lengths_context:train_context_lengts[:,i], 
                    rnn_input_context: train_context_padded[:,i], rnn_input_question: train_question_padded[:,i], answer:answer_data})
            if i % 100 == 0:
                print('Epoch:',e,'Step: ', i , ' loss: ', train_loss)

































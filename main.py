import tensorflow as tf
import numpy as np
import argparse
import os
import random
from nltk.tokenize.moses import MosesTokenizer

import utils
from models import LSTM

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

embedding_size = 50
hidden_state_size= 100
hidden_state_size_question = 100
batch_size = 1
learning_rate = 2e-3
epochs = 1

# Load the training / testing data.
train, test = utils.load_data(args.task, args.task, args.data_dir, word2id, tokenizer, args.batch_size, args.seperate_context)
train_examples_num = len(train)
max_length = utils.get_max_length(train + test)





for i in range(10):
    #Define placeholder
    lengths_question = tf.placeholder(dtype=tf.int32, shape=[None])
    lengths_context = tf.placeholder(dtype=tf.int32, shape=[None])
    answer = tf.placeholder(dtype=tf.int32, shape=[None])
    rnn_input_context = tf.placeholder(tf.int32, shape=[None, max_length])
    rnn_input_question = tf.placeholder(tf.int32, shape=[None, max_length])
    model = LSTM(context=rnn_input_context, context_length = lengths_context,
            question= rnn_input_question, question_length=lengths_question, answer=answer,
            vocab_size=vocab_size,
            optimizer =tf.train.AdamOptimizer(learning_rate),embedding_size_context=embedding_size, 
            embedding_size_question= embedding_size, hidden_layer_size=hidden_state_size,
            dropout = 0.3, recurrent_cell = tf.nn.rnn_cell.GRUCell )

    logits = model.inference
    loss = model.loss
    accuracy = model.accuracy
    train_op = model.train

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            i = 0
            for context_batch, question_batch, answer_batch in utils.batch_generator(train, batch_size, max_length):
                i +=batch_size
                context, context_lengths= context_batch
                question, question_lengths = question_batch

                train_loss,  o = sess.run([loss, train_op] ,
                        feed_dict={lengths_question:question_lengths, lengths_context:context_lengths, 
                        rnn_input_context: context, rnn_input_question:question, answer:answer_batch})

                if i % 500 == 0:
                    print('step: ',i, 'train_loss:', train_loss)

            #After each epoch print the test set
            t_context_batch, t_question_batch, t_answer_batch =  next(utils.batch_generator(test, 500, max_length))
            t_context, t_context_lengths = t_context_batch
            t_question, t_question_lengths = t_question_batch

            test_loss, t_accuracy, t_result = sess.run([ loss, accuracy, logits] ,
                feed_dict={lengths_question:t_question_lengths, lengths_context:t_context_lengths, 
                rnn_input_context: t_context, rnn_input_question:t_question, answer:t_answer_batch})
            print(' ------------ Epoch: ',e,'accuracy:',t_accuracy, 'test_loss:', test_loss, '------------')


        print('****Accuracy is', t_accuracy, 'now qualitative_inspection:\n\n' ) 
    tf.reset_default_graph()
    # utils.qualitative_inspection(t_context, t_context_lengths, t_question, t_question_lengths, 
            # t_answer_batch, t_result,id2word, max_evaluations=40)



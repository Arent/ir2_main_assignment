from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nltk.tokenize.moses import MosesTokenizer

from lstm import LSTM, LSTM_Config

import numpy as np
import tensorflow as tf

import argparse
import os
import utils 

def set_config(config, args):
    config.batch_size = args.batch_size

    return config

def main(args):
    config = set_config(LSTM_Config(), args)

    # Load the vocabulary.
    w2i, i2w = utils.load_vocab(args.vocab)
    config.vocab_size = len(w2i)
    tokenizer = MosesTokenizer()
    print("Vocabulary size: %d" % config.vocab_size)
    
    # Obtain test and training sets.
    train, test = utils.load_data(args.task, args.task, args.data_dir, w2i, tokenizer, config.batch_size, args.seperate_context)

    train_question, train_answer = list(zip(*train))
    test_question, test_answer = list(zip(*test))

    train_question, train_m_length = utils.pad_results(train_question, 66)
    test_question, test_m_length = utils.pad_results(test_question, 66)

    # Create the computational graph for the lstm
    lstm = LSTM(config)

    init = tf.global_variables_initializer()

    
    with tf.Session() as sess:
        sess.run(init)
        # sess.run(tf.tables_initializer())
        # sess.run(train.initializer)
        train_feed_dict = {lstm.context_input: train_question.T,
                           lstm.answer_labels: train_answer,
                           lstm.question_length: train_m_length} 

        test_feed_dict = {lstm.context_input: test_question.T,
                           lstm.answer_labels: test_answer,
                           lstm.question_length: test_m_length}

        for i in range(100):
            loss, _, acc = sess.run([lstm.loss_op, lstm.train_op, lstm.accuracy_op], feed_dict=train_feed_dict)
            # print("train loss", loss)
            # print("train acc", acc)
            
            predict, acc = sess.run([lstm.prediction, lstm.accuracy_op], feed_dict=test_feed_dict)

            print("test acc", acc)
        # print(loss)


if __name__ == '__main__':
    # Parse cmd line args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/en",
                        help="Directory containing the data.")
    parser.add_argument("--vocab", type=str, default="data/vocab.txt",
                        help="Vocabulary file")
    parser.add_argument("--task", type=str, default="1",
                        help="Task number")
    parser.add_argument("--batch_size", type=int, default=80,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate of the optimizer.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory to store the model parameters.")
    parser.add_argument('--seperate_context', type=bool, default=False,
                        help='seperate the context from the question')
    args = parser.parse_args()

    if args.data_dir is None or args.model_dir is None:
      print("--data_dir and/or --model_dir argument missing.")

    main(args)
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
	tf.reset_default_graph()
	config = set_config(LSTM_Config(), args)

	lstm = LSTM(config)

	init = tf.global_variables_initializer()

	sess.run()

	# Load the vocabulary.
	w2i, i2w = utils.load_vocab(args.vocab)
	vocab_size = len(w2i)
	tokenizer = MosesTokenizer()
	print("Vocabulary size: %d" % vocab_size)
	
	# Obtain test and training sets.
	train, test = utils.load_data(args.task, args.data_dir, args.vocab, tokenizer, config.batch_size)
	
	with tf.Session() as sess:
		_, loss = sess.run([lstm.train_op, lstm.train_op], feed_dict={mlp.labels: labels,
                                               mlp.x_in: x_in})

		print(loss)


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
	parser.add_argument("--learning_rate", type=float, default=0.0001,
	                    help="Learning rate of the optimizer.")
	parser.add_argument("--model_dir", type=str, default=None,
	                    help="Directory to store the model parameters.")
	args = parser.parse_args()

	if args.data_dir is None or args.model_dir is None:
	  print("--data_dir and/or --model_dir argument missing.")

	main(args)
import tensorflow as tf
import numpy as np
import argparse
import os

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
args = parser.parse_args()

# Load the vocabulary.
w2i, i2w = utils.load_vocab(args.vocab)
vocab_size = len(w2i)
tokenizer = MosesTokenizer()

# Load the training / testing data.
train, test = utils.load_data(args.task, args.task, args.data_dir, w2i, tokenizer, args.batch_size)

# Print the first training sample.
x, y = next(train)
x_ = [i2w[i] for i in x]
y_ = i2w[y]
print("Question: %s = %s" % (x, x_))
print("Answer: %s = %s" % (y, y_))

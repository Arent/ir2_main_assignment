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
parser.add_argument('--seperate_context', type=bool, default=True,
                    help='Batch size')

args = parser.parse_args()

# Load the vocabulary.
word2id, id2word = utils.load_vocab(args.vocab)
vocab_size = len(word2id)
print('vocabulary_size', vocab_size)
tokenizer = MosesTokenizer()
embedding_size = 50
hidden_state_size_context = 100
hidden_state_size_question = 100
batch_size = 1


# Load the training / testing data.
train, test = utils.load_data(args.task, args.task, args.data_dir, word2id, tokenizer, args.batch_size, args.seperate_context)


train_context, train_question, train_answer  = list(zip(*train))

test_context, test_question, test_answer  = list(zip(*train))


max_sentence_length = max(len(l) for l in train_context + test_context +
		train_question + test_question + train_answer + test_answer)


print(max_sentence_length)
# Print the first training sample.

# for i in range(10):
# 	context_ids, question_ids, answer_ids = train[i]

# 	context_str =  " ".join([id2word[w_id] for w_id in context_ids])
# 	question_str = " ".join([id2word[w_id] for w_id in question_ids])
# 	answer_str = " ".join([id2word[w_id] for w_id in answer_ids])

# 	print("Context: %s = %s \n" % (context_ids, context_str))
# 	print("Question: %s = %s \n" % (question_ids, question_str))
# 	print("Answer: %s = %s \n" % (answer_ids, answer_str))





# lengths_question = tf.placeholder(dtype=tf.int32, shape=[batch])
# rnn_input_context = tf.placeholder(tf.float32, [batch_size, max_sentence_length])
# rnn_input_question = tf.placeholder(tf.float32, [batch_size, max_sentence_length])

# # embedding_matrix_context = tf.get_variable("embedding_matrix_context", \
# #         [vocab_size, embedding_size], dtype=tf.float32)
# # embeddings_context = tf.nn.embedding_lookup(embedding_matrix, rnn_input_context) # [1, time, emb_size]

# # embedding_matrix_question = tf.get_variable("embedding_matrix_question", \
# #         [vocab_size, embedding_size], dtype=tf.float32)
# # embeddings_question = tf.nn.embedding_lookup(embedding_matrix, rnn_input_question) # [1, time, emb_size]


# # # Build RNN cell
# # lstm_context = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_size_context)
# # lstm_question = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_size_question)
	
# # # Run Dynamic RNN encoder_outpus: [max_time, batch_size, num_units] encoder_state: [batch_size, num_units]


# # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
# #     encoder_cell, encoder_emb_inp,

# #     sequence_length=source_sequence_length, time_major=True)






# import numpy as np
# import tensorflow as tf

# batch = 2
# dim = 3
# hidden = 4




# lengths = tf.placeholder(dtype=tf.int32, shape=[batch])
# inputs = tf.placeholder(dtype=tf.float32, shape=[batch, None, dim])
# cell = tf.nn.rnn_cell.GRUCell(hidden)
# cell_state = cell.zero_state(batch, tf.float32)
# output, _ = tf.nn.dynamic_rnn(cell, inputs, lengths, initial_state=cell_state)





# inputs_ = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
#                     [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
#                     dtype=np.int32)
# lengths_ = np.asarray([3, 1], dtype=np.int32)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     output_ = sess.run(output, {inputs: inputs_, lengths: lengths_})
#     print(output_)











































import tensorflow as tf

import tensorflow as tf
import argparse
import os
import sys
import numpy as np

from nltk.tokenize.moses import MosesTokenizer

import data_utils_old as utils
import misc_utils, eval_utils

from abcnn import ABCNN

parser = argparse.ArgumentParser()

# Todo check which arguments are needed
# Task info arguments.
parser.add_argument("--task", type=str, default="1",
                    help="Task number")

# I/O arguments.
parser.add_argument("--data_dir", type=str, default=None,
                    help="Directory containing the data.")
parser.add_argument("--vocab", type=str, default=None,
                    help="Vocabulary file")
parser.add_argument("--model_dir", type=str, default=None,
                    help="Directory to store the model parameters.")

# Training details arguments.
parser.add_argument("--batch_size", type=int, default=1000,
                    help="Batch size")
parser.add_argument("--optimizer", type=str, default="adam",
                    help="sgd|adam|adagrad|rmsprop")
parser.add_argument("--learning_rate", type=float, default=0.001,
                    help="Learning rate of the optimizer.")
parser.add_argument("--num_epochs", type=int, default=20,
                    help="Number of training epochs.")
parser.add_argument("--max_gradient_norm", type=float, default=1.0,
                    help="Maximum norm for gradients.")
parser.add_argument("--dropout_keep_prob", type=float, default=0.7,
                    help="Dropout keep probability")

# Model encoder arguments.
parser.add_argument("--embedding_size", type=int, default=64,
                    help="Size of the word embeddings.")
parser.add_argument("--encoder_type", type=str, default="uni",
                    help="uni|bi")
parser.add_argument("--num_units", type=int, default=64,
                    help="Number of hidden units in the RNN encoder")
parser.add_argument("--num_enc_layers", type=int, default=1,
                    help="Number of encoder layers in the encoder")

# Model decoder arguments.
parser.add_argument("--merge_mode", type=str, default="concat",
                    help="sum|concat")
parser.add_argument("--num_output_hidden", type=str, default="256",
                    help="Number of units or the hidden layers, formatted as h1_size,h2_size,etc")

# Parse all arguments.
args = parser.parse_args()

# Make sure a data directory, vocabulary file and model dir have been provided.
if args.data_dir is None or args.model_dir is None or args.vocab is None:
  print("--data_dir and/or --model_dir and/or --vocab argument missing.")
  sys.exit(1)

# Parse the necessary strings to the correct format.
optimizer = misc_utils.parse_optimizer(args.optimizer)
num_output_hidden = misc_utils.parse_num_hidden(args.num_output_hidden)

# Load the vocabulary.
w2i, i2w = utils.load_vocab("./data/vocab.txt")
vocab_size = len(w2i)

# Load the training / testing data.
print("Loading data...")
tokenizer = MosesTokenizer()

# Load the training / testing data.
train, test = utils.load_data("3", "", "./tasks/en", w2i, tokenizer, args.batch_size, True)
train_context, train_question, train_answer = list(zip(*train))
test_context, test_question, test_answer = list(zip(*train))

max_c_length = max(len(l) for l in train_context + test_context)
max_q_length = max(len(l) for l in train_question + test_question)

train_context_padded, train_context_lengths = utils.pad_results(train_context, max_c_length)
train_question_padded, train_question_lengths = utils.pad_results(train_question, max_q_length)

test_context_padded, test_context_lengths = utils.pad_results(test_context, max_c_length)
test_question_padded, test_question_lengths = utils.pad_results(test_question, max_q_length)

# Create the training model.
print("Building model...")
with tf.variable_scope("ABCNN"):
  context = tf.placeholder(tf.int32, shape=[args.batch_size, max_c_length])
  question = tf.placeholder(tf.int32, shape=[args.batch_size, max_q_length])
  answer = tf.placeholder(tf.int32, shape=args.batch_size, name="labels")

  train_model = ABCNN(tf.contrib.learn.ModeKeys.TRAIN, vocab_size,
                      embedding_size=args.embedding_size, encoder_type=args.encoder_type,
                      keep_prob=args.dropout_keep_prob, merge_mode=args.merge_mode,
                      optimizer=optimizer, learning_rate=args.learning_rate)

  # Build the training model graph.
  encoded_context = train_model.create_encoder("context_encoder",
                                               context, k=3)
  encoded_question = train_model.create_encoder("question_encoder",
                                                question, k=3)
  attention_matrix = train_model.create_attention_matrix(encoded_context, encoded_question)
  logits = train_model.create_decoder(attention_matrix, max_c_length, max_q_length)
  train_loss = train_model.loss(logits, answer)
  train_acc = train_model.accuracy(logits, answer)
  train_op = train_model.train_step(train_loss)

# Create the testing model.
with tf.variable_scope("ABCNN", reuse=True):
  test_context = tf.placeholder(tf.int32, shape=[args.batch_size, max_c_length])
  test_question = tf.placeholder(tf.int32, shape=[args.batch_size, max_q_length])
  test_answer_placeholder = tf.placeholder(tf.int32, shape=args.batch_size, name="labels")

  test_model = ABCNN(tf.contrib.learn.ModeKeys.EVAL, vocab_size,
                     embedding_size=args.embedding_size, encoder_type=args.encoder_type,
                     keep_prob=args.dropout_keep_prob, merge_mode=args.merge_mode,
                     optimizer=optimizer, learning_rate=args.learning_rate)

  # Build the testing model graph.
  test_encoded_context = test_model.create_encoder("context_encoder", test_context, k=3)
  test_encoded_question = test_model.create_encoder("question_encoder", test_question, k=3)
  test_attention_matrix = train_model.create_attention_matrix(test_encoded_context, test_encoded_question)
  test_logits = test_model.create_decoder(test_attention_matrix, max_c_length, max_q_length)
  test_acc = test_model.accuracy(test_logits, test_answer_placeholder)

# Create Tensorboard summaries.
train_loss_summary = tf.summary.scalar("train_loss", train_loss)
train_acc_summary = tf.summary.scalar("train_acc", train_acc)
train_summaries = tf.summary.merge([train_loss_summary, train_acc_summary])
test_summaries = tf.summary.scalar("test_accuracy", test_acc)

# Parameter saver.
saver = tf.train.Saver()
steps_per_stats = 3

with tf.Session() as sess:
  print("Running initializers...")
  sess.run(tf.global_variables_initializer())

  # Create the summary writers.
  print("Creating summary writers...")
  train_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "train"), sess.graph)
  test_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "test"), sess.graph)

  # Bookkeeping stuff.
  epoch_num = 1
  total_step = 0
  step = 0
  count = 0
  train_accs = []

  # Evaluate before training.
  print("Performing evaluation before training...")
  predictions, acc, summary = sess.run([tf.argmax(test_logits, axis=-1), test_acc, test_summaries],
                                       feed_dict={test_context: test_context_padded[:args.batch_size],
                                                  test_question: test_question_padded[:args.batch_size],
                                                  test_answer_placeholder: np.asarray(test_answer[:args.batch_size]).T[
                                                    0]}
                                       )
  print("---- Qualitative Analysis")
  eval_utils.qualitative_analysis(test_context_padded[:args.batch_size],
                                  test_question_padded[:args.batch_size],
                                  np.asarray(test_answer[:args.batch_size]).T[0],
                                  test_context_lengths[:args.batch_size],
                                  test_question_lengths[:args.batch_size],
                                  predictions, i2w, k=3)
  print("Test accuracy: %f" % acc)
  test_writer.add_summary(summary, epoch_num)
  print("=========================")

  while epoch_num <= args.num_epochs:

    try:
      # Train on all batches for one epoch.
      _, loss, summary = sess.run([train_op, train_loss, train_summaries],
                                  feed_dict={context: train_context_padded[:args.batch_size],
                                             question: train_question_padded[:args.batch_size],
                                             answer: np.asarray(train_answer[:args.batch_size]).T[0]}
                                  )
      step += 1
      total_step += 1
      train_writer.add_summary(summary, total_step)

      # Print training statistics periodically.
      if step % steps_per_stats == 0:
        print("Train loss at step %d = %f" % (step, loss))

      if step % 10 == 0:
        # Evaluate before training.
        print("Performing evaluation before training...")
        predictions, acc, summary = sess.run([tf.argmax(test_logits, axis=-1), test_acc, test_summaries],
                                             feed_dict={test_context: test_context_padded[:args.batch_size],
                                                        test_question: test_question_padded[:args.batch_size],
                                                        test_answer_placeholder:
                                                          np.asarray(test_answer[:args.batch_size]).T[
                                                            0]}
                                             )
        print("---- Qualitative Analysis")
        eval_utils.qualitative_analysis(test_context_padded[:args.batch_size],
                                        test_question_padded[:args.batch_size],
                                        np.asarray(test_answer[:args.batch_size]).T[0],
                                        test_context_lengths[:args.batch_size],
                                        test_question_lengths[:args.batch_size],
                                        predictions, i2w, k=3)
        print("Test accuracy: %f" % acc)
        test_writer.add_summary(summary, epoch_num)
        print("=========================")

    except tf.errors.OutOfRangeError:
      print("==== Finshed epoch %d ====" % epoch_num)

      # Save model parameters. TODO Commented this out because it takes a lot of time and space.
      # save_path = saver.save(sess, os.path.join(args.model_dir, "model_epoch_%d.ckpt" % epoch_num))
      # print("Model checkpoint saved in %s" % save_path)

      # Evaluate the model on the test set.
      print("Evaluating model...")
      predictions, acc, summary = sess.run([tf.argmax(test_logits, axis=-1), test_acc, test_summaries],
                                           feed_dict={test_context: test_context_padded[:args.batch_size],
                                                      test_question: test_question_padded[:args.batch_size],
                                                      test_answer_placeholder:
                                                        np.asarray(test_answer[:args.batch_size]).T[
                                                          0]}
                                           )
      print("---- Qualitative Analysis")
      eval_utils.qualitative_analysis(test_context_padded[:args.batch_size],
                                      test_question_padded[:args.batch_size],
                                      np.asarray(test_answer[:args.batch_size]).T[0],
                                      test_context_lengths[:args.batch_size],
                                      test_question_lengths[:args.batch_size],
                                      predictions, i2w, k=3)
      print("Test accuracy: %f" % acc)
      test_writer.add_summary(summary, epoch_num)
      print("=========================")

      # Re-initialize the training iterator.
      epoch_num += 1
      step = 0
      continue

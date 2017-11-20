import tensorflow as tf
import numpy as np
import argparse
import random
import os
import sys

from nltk.tokenize.moses import MosesTokenizer

import utils

from qarnn import QARNN

OPTIMIZER_DICT = {
                  "sgd": tf.train.GradientDescentOptimizer, # Gradient Descent
                  "adadelta": tf.train.AdagradDAOptimizer, # Adadelta
                  "adagrad": tf.train.AdagradOptimizer, # Adagrad
                  "adam": tf.train.AdamOptimizer, # Adam
                  "rmsprop": tf.train.RMSPropOptimizer # RMSprop
                  }

def qualitative_analysis(contexts, questions, answers, context_lengths,
                         question_lengths, predictions, i2w, k=1):
  size = questions.shape[0]
  indices = random.sample(range(size), k)
  for idx in indices:
    context = contexts[idx][:context_lengths[idx]]
    question = questions[idx][:question_lengths[idx]]
    answer = answers[idx]
    prediction = predictions[idx]
    context_ = " ".join(i2w[word_id] for word_id in context)
    question_ = " ".join(i2w[word_id] for word_id in question)
    answer_ = i2w[answer]
    prediction_ = i2w[prediction]
    print("Context: %s" % context_)
    print("Question: %s" % question_)
    print("Answer: %s" % answer_)
    print("Model prediction: %s" % prediction_)
    print("-------------------------")

def parse_cell_type(cell_type):
  if cell_type == "gru":
    return tf.contrib.rnn.GRUCell
  elif cell_type == "lstm":
    return tf.contrib.rnn.LSTMCell
  else:
    print("ERROR: unknown cell type")
    return None

# Parse cmd line args.
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None,
                    help="Directory containing the data.")
parser.add_argument("--vocab", type=str, default=None,
                    help="Vocabulary file")
parser.add_argument("--task", type=str, default="1",
                    help="Task number")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size")
parser.add_argument("--num_epochs", type=int, default=20,
                    help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=0.001,
                    help="Learning rate of the optimizer.")
parser.add_argument("--max_gradient_norm", type=float, default=1.0,
                    help="Maximum norm for gradients.")
parser.add_argument("--model_dir", type=str, default=None,
                    help="Directory to store the model parameters.")
parser.add_argument("--cell_type", type=str, default="lstm",
                    help="Cell type for the RNN.")
parser.add_argument("--num_units", type=int, default=64,
                    help="Number of hidden units in the RNN encoder")
parser.add_argument("--embedding_size", type=int, default=64,
                    help="Size of the word embeddings.")
parser.add_argument("--encoder_type", type=str, default="uni",
                    help="uni|bi")
parser.add_argument("--dropout_keep_prob", type=float, default=0.7,
                    help="Dropout keep probability")
parser.add_argument("--num_output_hidden", type=str, default="256",
                    help="Number of units or the hidden layers, formatted as h1_size,h2_size,etc")
parser.add_argument("--merge_mode", type=str, default="concat",
                    help="sum|concat")
parser.add_argument("--num_enc_layers", type=int, default=1,
                    help="Number of encoder layers in the encoder")
parser.add_argument("--optimizer", type=str, default="adam",
                    help="sgd|adam|adagrad|rmsprop")
args = parser.parse_args()

if args.data_dir is None or args.model_dir is None:
  print("--data_dir and/or --model_dir argument missing.")
  sys.exit(1)

optimizer = OPTIMIZER_DICT[args.optimizer]

if args.num_output_hidden:
  num_output_hidden = args.num_output_hidden.split(",")
  num_output_hidden = [int(num_hidden) for num_hidden in num_output_hidden]
else:
  num_output_hidden = []

cell_type = parse_cell_type(args.cell_type)

# Load the vocabulary.
w2i, i2w = utils.load_vocab(args.vocab)
vocab_size = len(w2i)
tokenizer = MosesTokenizer()
print("Vocabulary size: %d" % vocab_size)

# Load the training / testing data.
train, test = utils.load_data(args.task, args.data_dir, args.vocab, tokenizer, args.batch_size)
context, question, answer, context_length, question_length = train.get_next()
test_context, test_question, test_answer, test_context_length, test_question_length = test.get_next()

# Create the training model.
with tf.variable_scope("root"):
  train_model = QARNN(tf.contrib.learn.ModeKeys.TRAIN, vocab_size,
      embedding_size=args.embedding_size, num_units=args.num_units,
      encoder_type=args.encoder_type, keep_prob=args.dropout_keep_prob,
      cell_type=cell_type, num_output_hidden=num_output_hidden,
      num_enc_layers=args.num_enc_layers, merge_mode=args.merge_mode,
      optimizer=optimizer, learning_rate=args.learning_rate,
      max_gradient_norm=args.max_gradient_norm)

  # Build the training model graph.
  encoded_context = train_model.create_encoder("context_encoder",
      context, context_length)
  encoded_question = train_model.create_encoder("question_encoder",
      question, question_length)
  merged_encoding = train_model.merge_encodings(encoded_context,
      encoded_question)
  logits = train_model.create_decoder(merged_encoding)
  train_loss = train_model.loss(logits, answer)
  train_acc = train_model.accuracy(logits, answer)
  train_op = train_model.train_step(train_loss)

# Create the testing model.
with tf.variable_scope("root", reuse=True):
  test_model = QARNN(tf.contrib.learn.ModeKeys.EVAL, vocab_size,
      embedding_size=args.embedding_size, num_units=args.num_units,
      encoder_type=args.encoder_type, keep_prob=args.dropout_keep_prob,
      cell_type=cell_type, num_output_hidden=num_output_hidden,
      num_enc_layers=args.num_enc_layers, merge_mode=args.merge_mode,
      optimizer=optimizer, learning_rate=args.learning_rate,
      max_gradient_norm=args.max_gradient_norm)

  # Build the testing model graph.
  test_encoded_context = test_model.create_encoder("context_encoder",
      test_context, test_context_length)
  test_encoded_question = test_model.create_encoder("question_encoder",
      test_question, test_question_length)
  test_merged_encoding = test_model.merge_encodings(test_encoded_context,
      test_encoded_question)
  test_logits = test_model.create_decoder(test_merged_encoding)
  test_acc = test_model.accuracy(test_logits, test_answer)

# Create Tensorboard summaries.
train_loss_summary = tf.summary.scalar("train_loss", train_loss)
train_acc_summary = tf.summary.scalar("train_acc", train_acc)
train_summaries = tf.summary.merge([train_loss_summary, train_acc_summary])

test_summaries = tf.summary.scalar("test_accuracy", test_acc)

# Parameter saver.
saver = tf.train.Saver()
steps_per_stats = 3

# Train the model.
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  sess.run(train.initializer)

  # Create the summary writers.
  train_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "train"), sess.graph)
  test_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "test"), sess.graph)

  # Bookkeeping stuff.
  epoch_num = 0
  total_step = 0
  step = 0
  count = 0
  train_accs = []

  # Evaluate before training.
  sess.run(test.initializer)
  contexts, questions, answers, context_lengths, question_lengths, predictions, acc, summary = sess.run([
      test_context, test_question, test_answer, test_context_length, test_question_length,
      tf.argmax(test_logits, axis=-1), test_acc, test_summaries])
  print("---- Qualitative Analysis")
  qualitative_analysis(contexts, questions, answers, context_lengths, question_lengths, predictions, i2w, k=3)
  print("Test accuracy: %f" % acc)
  test_writer.add_summary(summary, epoch_num)
  print("=========================")

  while epoch_num < args.num_epochs:

    try:
      # Train on all batches for one epoch.
      _, loss, summary = sess.run([train_op, train_loss, train_summaries])
      step += 1
      total_step += 1
      train_writer.add_summary(summary, total_step)

      # Print training statistics periodically.
      if step % steps_per_stats == 0:
        print("Train loss at step %d = %f" % (step, loss))

    except tf.errors.OutOfRangeError:
      print("==== Finshed epoch %d ====" % epoch_num)

      # Save model parameters.
      # save_path = saver.save(sess, os.path.join(args.model_dir, "model_epoch_%d.ckpt" % epoch_num)) TODO
      # print("Model checkpoint saved in %s" % save_path)

      # Evaluate the model on the test set.
      sess.run(test.initializer)
      contexts, questions, answers, context_lengths, question_lengths, predictions, acc, summary = sess.run([
          test_context, test_question, test_answer, test_context_length, test_question_length,
          tf.argmax(test_logits, axis=-1), test_acc, test_summaries])
      print("---- Qualitative Analysis")
      qualitative_analysis(contexts, questions, answers, context_lengths, question_lengths, predictions, i2w, k=3)
      print("Test accuracy: %f" % acc)
      test_writer.add_summary(summary, epoch_num)
      print("=========================")

      # Re-initialize the training iterator.
      sess.run(train.initializer)
      epoch_num += 1
      step = 0
      continue

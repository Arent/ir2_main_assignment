import tensorflow as tf
import argparse
import os
import sys

from nltk.tokenize.moses import MosesTokenizer

import data_utils, eval_utils, misc_utils

from abcnn_haitam import ABCNN

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
parser.add_argument("--batch_size", type=int, default=8,
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

# Load the vocabulary into numpy for evaluation and debugging purposes.
w2i, i2w = data_utils.load_vocab(args.vocab)
vocab_size = len(w2i)

# Print the provided arguments for the user.
misc_utils.print_args(args)

# Load the training / testing data.
print("Loading data...")
tokenizer = MosesTokenizer()
train, test = data_utils.load_data(args.task, args.data_dir,
    args.vocab, tokenizer, args.batch_size)
context, question, answer, context_length, question_length = train.get_next()
test_context, test_question, test_answer, test_context_length, test_question_length = test.get_next()
print("vocabulary size = %d" % vocab_size)

# Create the training model.
print("Building model...")
with tf.variable_scope("ABCNN"):
  train_model = ABCNN(tf.contrib.learn.ModeKeys.TRAIN, vocab_size,
      embedding_size=args.embedding_size, encoder_type=args.encoder_type,
      keep_prob=args.dropout_keep_prob, merge_mode=args.merge_mode,
      optimizer=optimizer, learning_rate=args.learning_rate)

  # Build the training model graph.
  encoded_context = train_model.create_encoder("context_encoder",
      context, context_length, k=3)
  encoded_question = train_model.create_encoder("question_encoder",
      question, question_length, k = 3)
  attention_matrix = train_model.create_attention_matrix(encoded_context, encoded_question)
  logits = train_model.create_decoder(attention_matrix)
  train_loss = train_model.loss(logits, answer)
  train_acc = train_model.accuracy(logits, answer)
  train_op = train_model.train_step(train_loss)

# Create Tensorboard summaries.
train_loss_summary = tf.summary.scalar("train_loss", train_loss)
train_acc_summary = tf.summary.scalar("train_acc", train_acc)
train_summaries = tf.summary.merge([train_loss_summary, train_acc_summary])

# Parameter saver.
saver = tf.train.Saver()
steps_per_stats = 3

# Train the model.
with tf.Session() as sess:
  print("Running initializers...")
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  sess.run(train.initializer)

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

  while epoch_num <= args.num_epochs:

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

      # Save model parameters. TODO Commented this out because it takes a lot of time and space.
      # save_path = saver.save(sess, os.path.join(args.model_dir, "model_epoch_%d.ckpt" % epoch_num))
      # print("Model checkpoint saved in %s" % save_path)

      # Evaluate the model on the test set.
      print("Evaluating model...")
      sess.run(test.initializer)
      contexts, questions, answers, context_lengths, question_lengths, predictions, acc, summary = sess.run([
          test_context, test_question, test_answer, test_context_length, test_question_length,
          tf.argmax(logits, axis=-1), train_acc, train_summaries])
      print("---- Qualitative Analysis")
      eval_utils.qualitative_analysis(contexts, questions, answers, context_lengths, question_lengths,
          predictions, i2w, k=3)
      print("Test accuracy: %f" % acc)
      test_writer.add_summary(summary, epoch_num)
      print("=========================")

      # Re-initialize the training iterator.
      sess.run(train.initializer)
      epoch_num += 1
      step = 0
      continue
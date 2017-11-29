import tensorflow as tf
import argparse
import os
import sys

from nltk.tokenize.moses import MosesTokenizer

import data_utils, eval_utils, misc_utils
from qarnn import QARNN


import time
stamp = time.strftime('#%a_%H_%M_%S')

parser = argparse.ArgumentParser()

# Task info arguments.
parser.add_argument("--task", type=str, default="1",
                    help="Task number")

# I/O arguments.
parser.add_argument("--data_dir", type=str, default='data/baBI-tasks/en',
                    help="Directory containing the data.")
parser.add_argument("--vocab", type=str, default='data/vocab.txt',
                    help="Vocabulary file")
parser.add_argument("--model_dir", type=str, default='log/rand',
                    help="Directory to store the model parameters.")

# Training details arguments.
parser.add_argument("--batch_size", type=int, default=10,
                    help="Batch size")
parser.add_argument("--optimizer", type=str, default="adam",
                    help="sgd|adam|adagrad|rmsprop")
parser.add_argument("--learning_rate", type=float, default=0.0005,
                    help="Learning rate of the optimizer.")
parser.add_argument("--num_epochs", type=int, default=25,
                    help="Number of training epochs.")
parser.add_argument("--max_gradient_norm", type=float, default=1.0,
                    help="Maximum norm for gradients.")
parser.add_argument("--dropout_keep_prob", type=float, default=0.7,
                    help="Dropout keep probability")

# Model encoder arguments.
parser.add_argument("--embedding_size", type=int, default=50,
                    help="Size of the word embeddings.")
parser.add_argument("--cell_type", type=str, default="lstm",
                    help="Cell type for the RNN.")
parser.add_argument("--encoder_type", type=str, default="uni",
                    help="uni|bi")
parser.add_argument("--num_units", type=int, default=50,
                    help="Number of hidden units in the RNN encoder")
parser.add_argument("--num_enc_layers", type=int, default=1,
                    help="Number of encoder layers in the encoder")
parser.add_argument("--num_output_hidden", type=str, default="100",
                    help="Number of units or the hidden layers, formatted as h1_size,h2_size,etc")

# Model decoder arguments.
parser.add_argument("--merge_mode", type=str, default="concat",
                    help="sum|concat")
parser.add_argument("--use_attention", type=bool, default=False,
                    help="Wether to use an attention mechanism for predicting the output")

# Parse all arguments.
args = parser.parse_args()

# Make sure a data directory, vocabulary file and model dir have been provided.
if args.data_dir is None or args.model_dir is None or args.vocab is None:
  print("--data_dir and/or --model_dir and/or --vocab argument missing.")
  sys.exit(1)

# Parse the necessary strings to the correct format.
cell_type = misc_utils.parse_cell_type(args.cell_type)
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
with tf.variable_scope("QARNN"):
  train_model = QARNN(tf.contrib.learn.ModeKeys.TRAIN, vocab_size,
      embedding_size=args.embedding_size, num_units=args.num_units,
      encoder_type=args.encoder_type, keep_prob=args.dropout_keep_prob,
      cell_type=cell_type, num_output_hidden=num_output_hidden,
      num_enc_layers=args.num_enc_layers, merge_mode=args.merge_mode,
      optimizer=optimizer, learning_rate=args.learning_rate,
      max_gradient_norm=args.max_gradient_norm,
      use_attention=args.use_attention)


  # Build the training model graph.

  # question_context_question = tf.concat(question, context, axis=1)
  if args.use_attention:

    output_context, encoded_context = train_model.create_encoder("context_encoder",
        context, context_length)
    output_question, encoded_question = train_model.create_encoder("question_encoder",
        question, question_length)

    merged_encoding = train_model.create_attention_layer(encoded_question, output_context, context_length)

  else:
    encoded_context = train_model.create_encoder("context_encoder",
        context, context_length)
    encoded_question = train_model.create_encoder("question_encoder",
        question, question_length)

    merged_encoding = train_model.merge_encodings( encoded_context , encoded_question)

  logits = train_model.create_decoder(merged_encoding)
  train_loss = train_model.loss(logits, answer)
  train_acc = train_model.accuracy(logits, answer)
  train_op = train_model.train_step(train_loss)

# Create the testing model.
with tf.variable_scope("QARNN", reuse=True):
  test_model = QARNN(tf.contrib.learn.ModeKeys.EVAL, vocab_size,
      embedding_size=args.embedding_size, num_units=args.num_units,
      encoder_type=args.encoder_type, keep_prob=args.dropout_keep_prob,
      cell_type=cell_type, num_output_hidden=num_output_hidden,
      num_enc_layers=args.num_enc_layers, merge_mode=args.merge_mode,
      optimizer=optimizer, learning_rate=args.learning_rate,
      max_gradient_norm=args.max_gradient_norm,
      use_attention=args.use_attention)

  # Build the testing model graph.

  if args.use_attention:
    test_output_context, test_encoded_context = test_model.create_encoder("context_encoder",
        test_context, test_context_length)
    ttest_output_question, test_encoded_question = test_model.create_encoder("question_encoder",
        test_question, test_question_length)

    test_merged_encoding = test_model.create_attention_layer(test_encoded_question, test_output_context, test_context_length)
  
  else:
    test_encoded_context = test_model.create_encoder("context_encoder",
                          test_context, test_context_length)      
    test_encoded_question = test_model.create_encoder("question_encoder",
                          test_question, test_question_length)

    test_merged_encoding = test_model.merge_encodings(test_encoded_context, test_encoded_question)

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

  # Evaluate before training.
  print("Performing evaluation before training...")
  sess.run(test.initializer)
  contexts, questions, answers, context_lengths, question_lengths, predictions, acc, summary = sess.run([
      test_context, test_question, test_answer, test_context_length, test_question_length,
      tf.argmax(test_logits, axis=-1), test_acc, test_summaries])
  print("---- Qualitative Analysis")
  eval_utils.qualitative_analysis(contexts, questions, answers, context_lengths, question_lengths,
      predictions, i2w, k=3)
  print("Test accuracy: %f" % acc)
  test_writer.add_summary(summary, epoch_num)
  print("=========================")

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
          tf.argmax(test_logits, axis=-1), test_acc, test_summaries])
      print("---- Qualitative Analysis")
      eval_utils.qualitative_analysis(contexts, questions, answers, context_lengths, question_lengths,
          predictions, i2w, k=15)
      print("Test accuracy: %f" % acc)
      test_writer.add_summary(summary, epoch_num)
      print("=========================")

      # Re-initialize the training iterator.
      sess.run(train.initializer)
      epoch_num += 1
      step = 0
      continue

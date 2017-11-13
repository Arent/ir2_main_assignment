import tensorflow as tf
import numpy as np
import argparse
import random
import os

from nltk.tokenize.moses import MosesTokenizer

import utils

from lstm import QALSTM

def qualitative_analysis(questions, answers, question_lengths, predictions,
                         i2w, k=1):
  size = questions.shape[0]
  indices = random.sample(range(size), k)
  for idx in indices:
    question = questions[idx][:question_lengths[idx]]
    answer = answers[idx]
    prediction = predictions[idx]
    question_ = " ".join(i2w[word_id] for word_id in question)
    answer_ = i2w[answer]
    prediction_ = i2w[prediction]
    print("Question: %s" % question_)
    print("Answer: %s" % answer_)
    print("Model prediction: %s" % prediction_)
    print("-------------------------")

# Parse cmd line args.
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None,
                    help="Directory containing the data.")
parser.add_argument("--vocab", type=str, default=None,
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
  os.exit()

# Load the vocabulary.
w2i, i2w = utils.load_vocab(args.vocab)
vocab_size = len(w2i)
tokenizer = MosesTokenizer()
print("Vocabulary size: %d" % vocab_size)

# Load the training / testing data.
train, test = utils.load_data(args.task, args.data_dir, args.vocab, tokenizer, args.batch_size)
questions, answers, question_lengths = train.get_next()
test_questions, test_answers, test_question_lengths = test.get_next()

# Create the training model.
with tf.variable_scope("root"):
  train_model = QALSTM(vocab_size, questions, answers, question_lengths)
  train_loss = train_model.loss()
  optimizer = tf.train.AdamOptimizer(args.learning_rate)
  train_op = optimizer.minimize(train_loss)

# Create the testing model.
with tf.variable_scope("root", reuse=True):
  test_model = QALSTM(vocab_size, test_questions, test_answers, test_question_lengths)
  test_acc = test_model.accuracy()

# Create Tensorboard summaries.
loss_summary = tf.summary.scalar("train_loss", train_loss)
acc_summary = tf.summary.scalar("test_accuracy", test_acc)

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
  questions, answers, question_lengths, predictions, acc, summary = sess.run([
      test_model.question, test_model.answer, test_model.question_length,
      tf.argmax(test_model.logits, axis=-1), test_acc, acc_summary])
  print("---- Qualitative Analysis")
  qualitative_analysis(questions, answers, question_lengths, predictions, i2w, k=3)
  print("Test accuracy: %f" % acc)
  test_writer.add_summary(summary, epoch_num)
  print("=========================")

  while epoch_num < args.num_epochs:

    try:
      # Train on all batches for one epoch.
      _, loss, summary = sess.run([train_op, train_loss, loss_summary])
      step += 1
      total_step += 1
      train_writer.add_summary(summary, total_step)

      # Print training statistics periodically.
      if step % steps_per_stats == 0:
        print("Train loss at step %d = %f" % (step, loss))

    except tf.errors.OutOfRangeError:
      print("==== Finshed epoch %d ====" % epoch_num)

      # Save model parameters.
      save_path = saver.save(sess, os.path.join(args.model_dir, "model_epoch_%d.ckpt" % epoch_num))
      print("Model checkpoint saved in %s" % save_path)

      # Evaluate the model on the test set.
      sess.run(test.initializer)
      questions, answers, question_lengths, predictions, acc, summary = sess.run([
          test_model.question, test_model.answer, test_model.question_length,
          tf.argmax(test_model.logits, axis=-1), test_acc, acc_summary])
      print("---- Qualitative Analysis")
      qualitative_analysis(questions, answers, question_lengths, predictions, i2w, k=3)
      print("Test accuracy: %f" % acc)
      test_writer.add_summary(summary, epoch_num)
      print("=========================")

      # Re-initialize the training iterator.
      sess.run(train.initializer)
      epoch_num += 1
      step = 0
      continue

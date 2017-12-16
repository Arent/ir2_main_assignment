import tensorflow as tf
import argparse
import os
import sys

from nltk.tokenize.moses import MosesTokenizer

import data_utils, eval_utils, misc_utils

from qarnn import QARNN

parser = argparse.ArgumentParser()

# Task info arguments.
parser.add_argument("--task", type=str, default="1",
                    help="Task number")

# I/O arguments.
parser.add_argument("--data_dir", type=str, default=None,
                    help="Directory containing the data.")
parser.add_argument("--val_split", type=float, default=0.1,
                    help="Percentage of the training data to use for validation")
parser.add_argument("--vocab", type=str, default=None,
                    help="Vocabulary file")
parser.add_argument("--model_dir", type=str, default=None,
                    help="Directory to store the model parameters.")
parser.add_argument("--restore_from_dir", type=str, default=None)

# Training details arguments.
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size")
parser.add_argument("--optimizer", type=str, default="adam",
                    help="sgd|adam|adagrad|rmsprop")
parser.add_argument("--learning_rate", type=float, default=0.0005,
                    help="Learning rate of the optimizer.")
parser.add_argument("--num_epochs", type=int, default=10,
                    help="Number of training epochs.")
parser.add_argument("--max_gradient_norm", type=float, default=1.0,
                    help="Maximum norm for gradients.")
parser.add_argument("--dropout_keep_prob", type=float, default=0.8,
                    help="Dropout keep probability")

# General model arguments.
parser.add_argument("--model_type", type=str, default="attention",
                    help="normal|attention")

# Model encoder arguments.
parser.add_argument("--embedding_size", type=int, default=64,
                    help="Size of the word embeddings.")
parser.add_argument("--cell_type", type=str, default="gru",
                    help="Cell type for the RNN.")
parser.add_argument("--encoder_type", type=str, default="bi",
                    help="uni|bi")
parser.add_argument("--num_units", type=int, default=64,
                    help="Number of hidden units in the RNN encoder")
parser.add_argument("--num_enc_layers", type=int, default=1,
                    help="Number of encoder layers in the encoder")

# Model decoder arguments.
parser.add_argument("--merge_mode", type=str, default="concat",
                    help="sum|concat")

# Parse all arguments.
args = parser.parse_args()

# Make sure a data directory, vocabulary file and model dir have been provided.
if args.data_dir is None or args.model_dir is None or args.vocab is None:
  print("--data_dir and/or --model_dir and/or --vocab argument missing.")
  sys.exit(1)

assert (args.model_type == "normal" or args.model_type == "attention")
attention = args.model_type == "attention"

# Parse the necessary strings to the correct format.
cell_type = misc_utils.parse_cell_type(args.cell_type)
optimizer = misc_utils.parse_optimizer(args.optimizer)

# Load the vocabulary into numpy for evaluation and debugging purposes.
w2i, i2w = data_utils.load_vocab(args.vocab)
vocab_size = len(w2i)

# Print the provided arguments for the user.
misc_utils.print_args(args)

# Load the training / testing data.
print("Loading data...")
tokenizer = MosesTokenizer()
train, val, test, tf_vocab = data_utils.load_data(task_id=args.task, data_dir=args.data_dir,
    vocab_file=args.vocab, tokenizer=tokenizer, batch_size=args.batch_size, val_split=args.val_split, q_in_context=True)



context, question, answer_input, answer_output, context_length, question_length, answer_length = train.get_next()
val_context, val_question, val_answer_input, val_answer_output, val_context_length, val_question_length, val_answer_length = val.get_next()
test_context, test_question, test_answer_input, test_answer_output, test_context_length, test_question_length, test_answer_length = test.get_next()
print("vocabulary size = %d" % vocab_size)

# Create the training model.
print("Building model...")
with tf.variable_scope("QARNN"):
  train_model = QARNN(tf.contrib.learn.ModeKeys.TRAIN, tf_vocab, vocab_size,
      embedding_size=args.embedding_size, num_units=args.num_units,
      encoder_type=args.encoder_type, keep_prob=args.dropout_keep_prob,
      cell_type=cell_type, num_enc_layers=args.num_enc_layers,
      merge_mode=args.merge_mode, optimizer=optimizer,
      learning_rate=args.learning_rate,
      max_gradient_norm=args.max_gradient_norm, attention=attention)

  # Build the training model graph.
  emb_context, emb_matrix = train_model.create_embeddings(context,
      name="enc_embedding_matrix")
  emb_question, _ = train_model.create_embeddings(question,
      embedding_matrix=emb_matrix)
  context_outputs, final_context_state = train_model.create_encoder("context_encoder",
      emb_context, context_length)
  _, final_question_state = train_model.create_encoder("question_encoder",
      emb_question, question_length)
  merged_state = train_model.merge_states(final_context_state,
      final_question_state)

  if args.model_type == "attention":
    initial_state = final_question_state
    attention_states = context_outputs
  else:
    initial_state = merged_state
    attention_states = None

  emb_answer, dec_emb_matrix = train_model.create_embeddings(answer_input,
      name="dec_embedding_matrix")
  logits = train_model.create_rnn_decoder(emb_answer, answer_length,
      initial_state, dec_emb_matrix, attention_states=attention_states)
  train_loss = train_model.loss(logits, answer_output, answer_length)
  train_ppl = train_model.perplexity(train_loss, tf.shape(logits)[0], answer_length)
  train_acc = train_model.accuracy(logits, answer_output, answer_length)
  train_op = train_model.train_step(train_loss)

# Create the validation model.
with tf.variable_scope("QARNN", reuse=True):
  val_model = QARNN(tf.contrib.learn.ModeKeys.EVAL, tf_vocab, vocab_size,
      embedding_size=args.embedding_size, num_units=args.num_units,
      encoder_type=args.encoder_type, keep_prob=args.dropout_keep_prob,
      cell_type=cell_type,num_enc_layers=args.num_enc_layers, merge_mode=args.merge_mode,
      optimizer=optimizer, learning_rate=args.learning_rate,
      max_gradient_norm=args.max_gradient_norm, attention=attention)

  # Build the validation model graph.
  val_emb_context, val_emb_matrix = val_model.create_embeddings(val_context,
      name="enc_embedding_matrix")
  val_emb_question, _ = val_model.create_embeddings(val_question,
      embedding_matrix=val_emb_matrix)
  val_context_outputs, val_final_context_state = val_model.create_encoder("context_encoder",
      val_emb_context, val_context_length)
  _, val_final_question_state = val_model.create_encoder("question_encoder",
      val_emb_question, val_question_length)
  val_merged_state = val_model.merge_states(val_final_context_state,
      val_final_question_state)

  if args.model_type == "attention":
    val_initial_state = val_final_question_state
    val_attention_states = val_context_outputs
  else:
    val_initial_state = val_merged_state
    val_attention_states = None

  val_emb_answer, val_dec_emb_matrix = train_model.create_embeddings(
      val_answer_input, name="dec_embedding_matrix")
  val_logits = val_model.create_rnn_decoder(val_emb_answer, val_answer_length,
      val_initial_state, val_dec_emb_matrix, attention_states=val_attention_states)
  val_loss = val_model.loss(val_logits, val_answer_output, val_answer_length)
  val_ppl = val_model.perplexity(val_loss, tf.shape(val_logits)[0], val_answer_length)
  val_acc = val_model.accuracy(val_logits, val_answer_output, val_answer_length)

# Create the testing model.
with tf.variable_scope("QARNN", reuse=True):
  test_model = QARNN(tf.contrib.learn.ModeKeys.INFER, tf_vocab, vocab_size,
      embedding_size=args.embedding_size, num_units=args.num_units,
      encoder_type=args.encoder_type, keep_prob=args.dropout_keep_prob,
      cell_type=cell_type,  num_enc_layers=args.num_enc_layers, merge_mode=args.merge_mode,
      optimizer=optimizer, learning_rate=args.learning_rate,
      max_gradient_norm=args.max_gradient_norm, attention=attention)

  # Build the testing model graph.
  test_emb_context, test_emb_matrix = test_model.create_embeddings(test_context,
      name="enc_embedding_matrix")
  test_emb_question, _ = test_model.create_embeddings(test_question,
      embedding_matrix=test_emb_matrix)
  test_context_outputs, test_final_context_state = test_model.create_encoder("context_encoder",
      test_emb_context, test_context_length)
  _, test_final_question_state = test_model.create_encoder("question_encoder",
      test_emb_question, test_question_length)
  test_merged_state = test_model.merge_states(test_final_context_state,
      test_final_question_state)

  if args.model_type == "attention":
    test_initial_state = test_final_question_state
    test_attention_states = test_context_outputs
  else:
    test_initial_state = test_merged_state
    test_attention_states = None

  test_emb_answer, test_dec_emb_matrix = test_model.create_embeddings(
      test_answer_input, name="dec_embedding_matrix")
  test_predictions, test_decoder_final_states = test_model.create_rnn_decoder(test_emb_answer, test_answer_length,
      test_initial_state, test_dec_emb_matrix, attention_states=test_attention_states)
  test_acc = tf.placeholder(tf.float32, shape=[])

# Create Tensorboard summaries.
train_summaries = tf.summary.scalar("train_loss", train_loss)
val_acc_summary = tf.summary.scalar("val_accuracy", val_acc)
val_ppl_summary = tf.summary.scalar("val_perplexity", val_ppl)
val_summaries = tf.summary.merge([val_acc_summary, val_ppl_summary])
test_summaries = tf.summary.scalar("test_accuracy", test_acc)


# Parameter saver.
saver = tf.train.Saver()
steps_per_stats = 100

# Train the model.
with tf.Session() as sess:
  print("Running initializers...")

  if args.restore_from_dir is not None:
    saver.restore(sess, tf.train.latest_checkpoint(args.restore_from_dir))
  else:
    sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  sess.run(train.initializer)

  # Create the summary writers.
  print("Creating summary writers...")
  train_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "train"), sess.graph)
  val_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "val"), sess.graph)
  test_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "test"), sess.graph)

  # Bookkeeping stuff.
  epoch_num = 1
  total_step = 0
  count = 0
  train_accs = []
  best_model = None

  # Evaluate before training.
  print("Performing evaluation before training...")
  sess.run(test.initializer)
  contexts, questions, answers, context_lengths, question_lengths, answer_lengths, predictions = sess.run([
      test_context, test_question, test_answer_output, test_context_length, test_question_length,
      test_answer_length, test_predictions])
  print("---- Qualitative Analysis")
  eval_utils.qualitative_analysis(contexts, questions, answers, context_lengths, question_lengths,
      answer_lengths, predictions, i2w, k=1)
  print("=========================")

  while epoch_num <= args.num_epochs:

    try:
      # Train on all batches for one epoch.
      _, loss, acc, ppl, summary = sess.run([train_op, train_loss, train_acc, train_ppl, train_summaries])
      total_step += 1
      train_writer.add_summary(summary, total_step)

      # Print training statistics periodically.
      if total_step % steps_per_stats == 0:
        print("Step %d: train_loss = %f, train_acc = %f, train ppl = %f" % (total_step, loss, acc, ppl))

    except tf.errors.OutOfRangeError:
      print("==== Finshed epoch %d ====" % epoch_num)

      # Save model parameters. TODO Commented this out because it takes a lot of time and space.
     # save_path = saver.save(sess, os.path.join(args.model_dir, "model_epoch_%d.ckpt" % epoch_num))
     # print("Model checkpoint saved in %s" % save_path)

      # Evaluate on the validation set.
      print("Evaluating model...")
      sess.run(val.initializer)
      val_accuracy, val_perplexity, summary =  sess.run([val_acc, val_ppl, val_summaries])
      val_writer.add_summary(summary, epoch_num)
      print("Epoch %d: validation accuracy = %f -- validation perplexity = %f" %
          (epoch_num, val_accuracy, val_perplexity))

      # Evaluate the model on the test set.
      sess.run(test.initializer)
      contexts, questions, answers, context_lengths, question_lengths, answer_lengths, predictions = sess.run([
          test_context, test_question, test_answer_output, test_context_length, test_question_length,
          test_answer_length, test_predictions])
      test_accuracy = eval_utils.compute_test_accuracy(answers, answer_lengths,
          predictions, w2i["</s>"], args.task != "8")
      print("Epoch %d: test accuracy: %f" % (epoch_num, test_accuracy))
      print("---- Qualitative Analysis")
      eval_utils.qualitative_analysis(contexts, questions, answers, context_lengths, question_lengths,
          answer_lengths, predictions, i2w, k=1)
      summary = sess.run(test_summaries, feed_dict={test_acc: test_accuracy})
      test_writer.add_summary(summary, epoch_num)
      print("=========================")

      # Save the best model.
      if best_model is None or best_model[2] < val_accuracy:
        best_model = (epoch_num, val_perplexity, val_accuracy, test_accuracy)

        # Save model parameters.
        save_path = saver.save(sess, os.path.join(args.model_dir, "best_model.ckpt"))
        print("Model checkpoint saved in %s" % save_path)

      # Re-initialize the training iterator.
      sess.run(train.initializer)
      epoch_num += 1
      continue

print("Best model (acc) at epoch %d -- validation perplexity = %f -- validation accuracy = %f -- test accuracy = %f" %
    best_model)
print(best_model[-2])

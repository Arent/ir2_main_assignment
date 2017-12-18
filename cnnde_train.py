import tensorflow as tf

import tensorflow as tf
import argparse
import os
import sys
import numpy as np

from nltk.tokenize.moses import MosesTokenizer

import data_utils_cnnde as utils
import misc_utils, eval_utils

from cnnde import ABCNN

parser = argparse.ArgumentParser()

# Task info arguments.
parser.add_argument("--task", type=str, default="1", help="Task number")

# I/O arguments.
parser.add_argument("--data_dir", type=str, default=None,
                    help="Directory containing the data.")
parser.add_argument("--vocab", type=str, default=None,
                    help="Vocabulary file")
parser.add_argument("--model_dir", type=str, default=None,
                    help="Directory to store the model parameters.")

# Training details arguments.
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--t_batch_size", type=int, default=1000,
                    help="Batch size of the test set")
parser.add_argument("--optimizer", type=str, default="adam",
                    help="sgd|adam|adagrad|rmsprop")
parser.add_argument("--learning_rate", type=float, default=0.001
                    ,
                    help="Learning rate of the optimizer.")
parser.add_argument("--num_epochs", type=int, default=500,
                    help="Number of training epochs.")
parser.add_argument("--eval_steps", type=int, default=1000,
                    help="Amount of steps after which should be evaluated.")
parser.add_argument("--dropout_keep_prob", type=float, default=0.8,
                    help="Dropout keep probability")
parser.add_argument("--decay_steps", type=float, default=100000,
                    help="Amount of batch steps untill the learning rate is multiplied by 96")

# Model encoder arguments.
parser.add_argument("--embedding_size", type=int, default=32,
                    help="Size of the word embeddings and conv output")
parser.add_argument("--num_layers", type=int, default=2,
                    help="Amount of convolutional layer blocks")
parser.add_argument("--kernel_size", type=int, default=16,
                    help="Width of the convolutional filter for the context encoding")

np.random.seed(42)
tf.set_random_seed(42)

# Parse all arguments.
args = parser.parse_args()

# Make sure a data directory, vocabulary file and model dir have been provided.
if args.data_dir is None or args.model_dir is None or args.vocab is None:
    print("--data_dir and/or --model_dir and/or --vocab argument missing.")
    sys.exit(1)

# Parse the necessary strings to the correct format.
optimizer = misc_utils.parse_optimizer(args.optimizer)

# Load the vocabulary.
w2i, i2w = utils.load_vocab("./data/vocab.txt")
vocab_size = len(w2i)

# Load the training / testing data.
print("Loading data...")
tokenizer = MosesTokenizer()

# Load the training / testing data.
train_val, test = utils.load_data(args.task, "", args.data_dir, w2i, tokenizer, args.batch_size, True)
train, val = train_val

train_context, train_question, train_answer = list(zip(*train))
val_context, val_question, val_answer = list(zip(*val))
test_context, test_question, test_answer = list(zip(*test))

print("Len of training set : {}, len of validation set : {}".format(len(train_context), len(val_context)))

max_c_length = max(len(l) for l in train_context + test_context + val_context)
avg_c_length = sum(len(l) for l in train_context + test_context + val_context) / len(train_context)
max_q_length = max(len(l) for l in train_question + test_question + val_question)

print("Average amount of words in context (used for normalization): {}".format(avg_c_length))

test_context_padded, test_context_lengths = utils.pad_results(test_context, max_c_length)
test_question_padded, test_question_lengths = utils.pad_results(test_question, max_q_length)

val_context_padded, val_context_lengths = utils.pad_results(val_context, max_c_length)
val_question_padded, val_question_lengths = utils.pad_results(val_question, max_q_length)

train_examples_num = len(train)

# Create the training model.
print("Building model...")
with tf.variable_scope("ABCNN"):
    context = tf.placeholder(tf.int32, shape=[args.batch_size, max_c_length])
    question = tf.placeholder(tf.int32, shape=[args.batch_size, max_q_length])
    answer = tf.placeholder(tf.int32, shape=args.batch_size, name="labels")

    train_model = ABCNN(vocab_size,
                        embedding_size=args.embedding_size,
                        keep_prob=args.dropout_keep_prob,
                        optimizer=optimizer,
                        learning_rate=args.learning_rate,
                        decay_steps=args.decay_steps)

    # Build the training model graph.
    e_context = train_model.create_embedding("context_encoder", context, max_c_length)
    e_question = train_model.create_embedding("question_encoder", question, max_q_length, True)
    encoded_context = train_model.create_encoder("context_encoder", e_context, k=args.kernel_size, num_layers=args.num_layers)
    encoded_question = train_model.create_encoder("question_encoder", e_question, k=5, num_layers=1)
    attention_matrix = train_model.create_attention_matrix(encoded_context, encoded_question)
    logits = train_model.create_decoder(attention_matrix, encoded_context, e_context, max_c_length, max_q_length,
                                        avg_c_length)
    train_loss = train_model.loss(logits, answer)
    train_acc = train_model.accuracy(logits, answer)
    global_step = tf.Variable(0, trainable=False)
    train_op = train_model.train_step(train_loss, global_step)

# Create the validation model.
with tf.variable_scope("ABCNN", reuse=True):
    val_context = tf.placeholder(tf.int32, shape=[args.t_batch_size, max_c_length])
    val_question = tf.placeholder(tf.int32, shape=[args.t_batch_size, max_q_length])
    val_answer_placeholder = tf.placeholder(tf.int32, shape=args.t_batch_size, name="labels")

    val_model = ABCNN(vocab_size,
                       embedding_size=args.embedding_size,
                       keep_prob=1.0,
                       optimizer=optimizer,
                       learning_rate=args.learning_rate,
                       decay_steps=args.decay_steps)

    # Build the testing model graph.
    val_e_context = val_model.create_embedding("context_encoder", val_context, max_c_length)
    val_e_question = val_model.create_embedding("question_encoder", val_question, max_q_length)
    val_encoded_context = val_model.create_encoder("context_encoder", val_e_context, k=args.kernel_size, num_layers=args.num_layers)
    val_encoded_question = val_model.create_encoder("question_encoder", val_e_question, k=5, num_layers=1)
    val_attention_matrix = val_model.create_attention_matrix(val_encoded_context, val_encoded_question)
    val_logits = val_model.create_decoder(val_attention_matrix, val_encoded_context, val_e_context, max_c_length,
                                            max_q_length, avg_c_length)
    val_loss = val_model.loss(val_logits, val_answer_placeholder)
    val_acc = val_model.accuracy(val_logits, val_answer_placeholder)

# Create the testing model.
with tf.variable_scope("ABCNN", reuse=True):
    test_context = tf.placeholder(tf.int32, shape=[args.t_batch_size, max_c_length])
    test_question = tf.placeholder(tf.int32, shape=[args.t_batch_size, max_q_length])
    test_answer_placeholder = tf.placeholder(tf.int32, shape=args.t_batch_size, name="labels")

    test_model = ABCNN(vocab_size,
                       embedding_size=args.embedding_size,
                       keep_prob=1.0,
                       optimizer=optimizer,
                       learning_rate=args.learning_rate,
                       num_layers=args.num_layers,
                       decay_steps=args.decay_steps)

    # Build the testing model graph.
    test_e_context = test_model.create_embedding("context_encoder", test_context, max_c_length)
    test_e_question = test_model.create_embedding("question_encoder", test_question, max_q_length)
    test_encoded_context = test_model.create_encoder("context_encoder", test_e_context, k=args.kernel_size, num_layers=args.num_layers)
    test_encoded_question = test_model.create_encoder("question_encoder", test_e_question, k=5, num_layers=1)
    test_attention_matrix = test_model.create_attention_matrix(test_encoded_context, test_encoded_question)
    test_logits = test_model.create_decoder(test_attention_matrix, test_encoded_context, test_e_context, max_c_length,
                                            max_q_length, avg_c_length)
    test_loss = test_model.loss(test_logits, test_answer_placeholder)
    test_acc = test_model.accuracy(test_logits, test_answer_placeholder)

# Create Tensorboard summaries.
train_loss_summary = tf.summary.scalar("train_loss", train_loss)
train_acc_summary = tf.summary.scalar("train_acc", train_acc)
train_summaries = tf.summary.merge([train_loss_summary, train_acc_summary])
test_summaries = tf.summary.scalar("test_accuracy", test_acc)
val_summaries = tf.summary.scalar("val_accuracy", val_acc)

# Parameter saver.
saver = tf.train.Saver()
steps_per_stats = 3


def test_analysis( sess, qualitative=True ):
    print("Evaluating model...")
    predictions, acc, l, summary = sess.run([tf.argmax(test_logits, axis=-1), test_acc, test_loss, test_summaries],
                                            feed_dict={ test_context: test_context_padded[:args.t_batch_size],
                                                        test_question: test_question_padded[:args.t_batch_size],
                                                        test_answer_placeholder:
                                                            np.asarray(test_answer[:args.t_batch_size]).T[0] }
                                            )
    v_predictions, v_acc, v_l, v_summary = sess.run([tf.argmax(val_logits, axis=-1), val_acc, val_loss, val_summaries],
                                            feed_dict={ val_context: val_context_padded,
                                                        val_question: val_question_padded,
                                                        val_answer_placeholder: np.asarray(val_answer).T[0] }
                                            )
    if (qualitative):
        print("---- Qualitative Analysis")
        eval_utils.qualitative_analysis(test_context_padded[:args.t_batch_size],
                                        test_question_padded[:args.t_batch_size],
                                        np.asarray(test_answer[:args.t_batch_size]).T[0],
                                        test_context_lengths[:args.t_batch_size],
                                        test_question_lengths[:args.t_batch_size],
                                        predictions, i2w, k=3)
    print("Test accuracy: {}, Test loss: {}".format(acc, l))
    print("Val accuracy: {}, Val loss: {}".format(v_acc, v_l))
    test_writer.add_summary(summary, total_step)
    val_writer.add_summary(summary, total_step)
    print("=========================")
    return v_acc, acc

with tf.Session() as sess:
    print("Running initializers...")
    sess.run(tf.global_variables_initializer())

    # Create the summary writers.
    print("Creating summary writers...")
    train_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "train"), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "val"), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(args.model_dir, "test"), sess.graph)

    print("Running task {}".format(args.task))

    # Bookkeeping stuff.
    total_step = 0.0
    best_test = 0.0
    best_val = 0.0
    best_e = 0.0
    stop = False

    # Evaluate before training.
    test_analysis(sess)
    for e in range(args.num_epochs):
        print("=========== EPOCH {} ============".format(e))
        print("=================================")
        i = 0

        for context_batch, question_batch, answer_batch in utils.batch_generator(train, args.batch_size, max_c_length,
                                                                                 max_q_length):
            i += args.batch_size
            # Train on all batches for one epoch.
            _, tr_loss, summary, tr_acc, g = sess.run([train_op, train_loss, train_summaries, train_acc, global_step],
                                                   feed_dict={ context: context_batch[0],
                                                               question: question_batch[0],
                                                               answer: np.asarray(answer_batch) })
            total_step += 1
            train_writer.add_summary(summary, total_step)

            # Print training statistics periodically.
            if i % steps_per_stats == 0:
                print("Epoch: {}; Step/batch_step: {},{}; Train accuracy: {}; Train loss: {}".format(e, i, g, tr_acc, tr_loss))

            # Save best results and or
            if i % args.eval_steps == 0:
                current_val_acc, current_test_acc = test_analysis(sess)

                if(current_val_acc >= best_val):
                    best_test = current_test_acc
                    best_val = current_val_acc
                    best_e = e
                print("The best model is at epoch {} with a test acc of {} and val acc of {}".format(best_e, best_test, best_val))

                if(current_test_acc == 1.0):
                    sys.exit(1)

        print("The best model is at epoch {} with a test acc of {} and val acc of {}".format(best_e, best_test, best_val))
        print("==== Finshed epoch %d ====" % e)

        test_analysis(sess)
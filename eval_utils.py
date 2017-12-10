import random
import numpy as np

"""
  Samples k random context-question-answer pairs and print them together
  with the model's predictions.
"""
def qualitative_analysis(contexts, questions, answers, sentence_lengths,
                         question_lengths, answer_lengths, predictions, i2w, k=1):
  size = questions.shape[0]
  indices = random.sample(range(size), k)
  for idx in indices:
    context = contexts[idx]
    sen_lens = sentence_lengths[idx]
    question = questions[idx][:question_lengths[idx]]
    answer = answers[idx][:answer_lengths[idx]]
    prediction = predictions[idx]
    context_ = " ".join([" ".join([i2w[word_id] for word_id in sent[:sen_lens[sent_id]]]) for sent_id, sent in enumerate(context)])
    question_ = " ".join(i2w[word_id] for word_id in question)
    answer_ = " ".join(i2w[word_id] for word_id in answer)
    prediction_ = " ".join(i2w[word_id] for word_id in prediction)
    print("Context: %s" % context_)
    print("Question: %s" % question_)
    print("Answer: %s" % answer_)
    print("Model prediction: %s" % prediction_)

"""
    Computes the test accuracy, not including the accuracy on end-of-sentence
    symbols and allowing for calculating accuracy when order doesn't matter.
"""
def compute_test_accuracy(answers, answer_lengths, predictions, eos_id, order_matters=True):

  # Cut off end-of-sentence symbols for the accuracy computation.
  answer_lengths = answer_lengths - 1

  # If the prediction size is smaller than the answer size, fill with -1s.
  shape_diff = answers.shape[-1] - predictions.shape[-1]
  if shape_diff > 0:
    predictions = np.pad(predictions, ((0, 0), (0, shape_diff)), "constant", constant_values=-1)

  # And cut off the end-of-sentence symbols for predictions as well.
  predictions = predictions[:, :answers.shape[-1]]

  # If order doesn't matter, sort the sequences
  if not order_matters:
    answers = np.sort(answers, axis=1)
    predictions = np.sort(predictions, axis=1)

  # Calculate accuracy, mask eos symbols.
  mask = (answers != eos_id).astype(np.float32)
  equals = np.equal(answers, predictions).astype(np.float32)
  masked_equals = mask * equals
  accuracies = np.sum(masked_equals, axis=1) / answer_lengths
  mean_accuracy = np.sum(accuracies) / float(len(accuracies))

  return mean_accuracy

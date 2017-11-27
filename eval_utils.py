import random

"""
  Samples k random context-question-answer pairs and print them together
  with the model's predictions.
"""
def qualitative_analysis(contexts, questions, answers, context_lengths,
                         question_lengths, answer_lengths, predictions, i2w, k=1):
  size = questions.shape[0]
  indices = random.sample(range(size), k)
  for idx in indices:
    context = contexts[idx][:context_lengths[idx]]
    question = questions[idx][:question_lengths[idx]]
    answer = answers[idx][:answer_lengths[idx]]
    prediction = predictions[idx][:answer_lengths[idx]] # TODO this function needs to change for when doing real decoding
    context_ = " ".join(i2w[word_id] for word_id in context)
    question_ = " ".join(i2w[word_id] for word_id in question)
    answer_ = " ".join(i2w[word_id] for word_id in answer)
    prediction_ = " ".join(i2w[word_id] for word_id in prediction)
    print("Context: %s" % context_)
    print("Question: %s" % question_)
    print("Answer: %s" % answer_)
    print("Model prediction: %s" % prediction_)

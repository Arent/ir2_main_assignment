import numpy as np

class Evaluation():
    """ Initialize a reusable evaluation class with the context/questions and answers. """
    def __init__(self, cntx_question, answers, id2word):
        self.id2word = id2word

        # Store the combination of questions and context.
        self.cntx_question = cntx_question

        # Answers consisting of id's
        self.label_answers = answers

        # self.questions = {}
        self.answers = {}
        self.predictions = {}

        self.predicted_answers = []

    """ Provide a list of predictions corresponding to all Q/A in this class."""
    def set_predictions(self, predictions):
        # Align the shapes of answers and predictions.
        self.predictions = predictions[:,np.newaxis]

        self.correct_indices = [i for i, (p, l) in enumerate(zip(predictions, self.label_answers)) if p == l]

        self.incorrect_indices = [i for i, (p, l) in enumerate(zip(predictions, self.label_answers)) if p != l]

    def print_predictions(self, amount=1, correct=False):
        if correct:
            if(len(self.correct_indices) <= 0):
                return            
            amount = min(amount, len(self.correct_indices))
            idx = np.random.choice(len(self.correct_indices), amount)
        else:
            if(len(self.incorrect_indices) <= 0):
                return
            amount = min(amount, len(self.incorrect_indices))
            idx = np.random.choice(len(self.incorrect_indices), amount)
        print(idx)
        for i in idx:
            print("The question:")
            print(self.cntx_question[i])
            print("The correct answer", self.id2word[self.label_answers[i][0]], 
                  "Given answer:", self.id2word[self.predictions[i][0]])

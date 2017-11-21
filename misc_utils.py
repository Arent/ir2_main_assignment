import tensorflow as tf

OPTIMIZER_DICT = {
                  "sgd": tf.train.GradientDescentOptimizer, # Gradient Descent
                  "adadelta": tf.train.AdagradDAOptimizer, # Adadelta
                  "adagrad": tf.train.AdagradOptimizer, # Adagrad
                  "adam": tf.train.AdamOptimizer, # Adam
                  "rmsprop": tf.train.RMSPropOptimizer # RMSprop
                  }

# Parses a text cell type and returns the corresponding initializer.
def parse_cell_type(cell_type_str):
  if cell_type_str == "gru":
    return tf.contrib.rnn.GRUCell
  elif cell_type_str == "lstm":
    return tf.contrib.rnn.LSTMCell
  else:
    print("ERROR: unknown cell type")
    return None

# Parses a text optimizer and returns the corresponding initializer.
def parse_optimizer(optimizer_str):
  return OPTIMIZER_DICT[optimizer_str]

# Parses a string formatted as h1_size,h2_size,etc. to an array of numbers.
def parse_num_hidden(num_hidden_str):
  if num_hidden_str:
    num_output_hidden = num_hidden_str.split(",")
    num_output_hidden = [int(num_hidden) for num_hidden in num_output_hidden]
  else:
    num_output_hidden = []
  return num_output_hidden

def print_args(args):
  for name, val in args.__dict__.items():
    print("%-20s %s" % (name + ":", val))

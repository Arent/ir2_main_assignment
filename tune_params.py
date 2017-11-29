import subprocess
import numpy as np
import argparse
import sys
import os

# Disable info logs from tensorflows.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

parser = argparse.ArgumentParser()

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

# Parse all arguments.
args = parser.parse_args()

# Make sure a data directory, vocabulary file and model dir have been provided.
if args.data_dir is None or args.model_dir is None or args.vocab is None:
  print("--data_dir and/or --model_dir and/or --vocab argument missing.")
  sys.exit(1)

def run_model(run_args):
  run_string = "python main.py --task %s --data_dir %s --model_dir %s --vocab %s %s" % \
      (args.task, args.data_dir, args.model_dir, args.vocab, run_args)
  output = subprocess.check_output([run_string], shell=True).decode('utf-8')
  return float(output.split()[-1])

learning_rates = [0.0001, 0.0005, 0.001, 0.005]
batch_sizes = [8, 16, 32, 64]
num_units = [32, 64, 128]
enc_types = [("bi", 1), ("uni", 1), ("uni", 2)]
emb_sizes = [16, 32, 64, 128]
dropout_keep_probs = [1.0, 0.8, 0.6, 0.4]
cell_types = ["gru", "lstm"]

best_lr = 0.001
best_batch_size = 8
best_num_units = 64
best_enc_type = ("bi", 1)
best_emb_size = 64
best_dropout_kp = 0.8
best_cell_type = "gru"

tunable_params = [learning_rates, batch_sizes, num_units, enc_types,
    emb_sizes, dropout_keep_probs, cell_types]
param_names = ["learning rate", "batch size", "num units",
    "encoder type", "embedding size", "dropout keep prob", "cell type"]
cur_setting = [best_lr, best_batch_size, best_num_units,
    best_enc_type, best_emb_size, best_dropout_kp, best_cell_type]
print("Current param settings: %s for %s\n" % (cur_setting, param_names))

best_total_score = -1.0
best_total_setting = None

# Repeat n times.
num_line_searches = 3
for _ in range(num_line_searches):

  # Do a line search over all params.
  for params_idx, params in enumerate(tunable_params):
    print("Tuning %s..." % param_names[params_idx])
    cur_params = list(cur_setting)
    cur_param_best_score = -1.0
    cur_best_param_val = None

    # Tune only this parameter, keep rest to best.
    for param in params:
      cur_params[params_idx] = param
      enc_type_args = "--encoder_type %s --num_enc_layers %d" % cur_params[3]
      run_args = ("--learning_rate %f --batch_size %d --num_units %d " + \
          "%s --embedding_size %d --dropout_keep_prob %f --cell_type %s") % \
          (cur_params[0], cur_params[1], cur_params[2], enc_type_args,
          cur_params[4], cur_params[5], cur_params[6])
      score = run_model(run_args)

      print("Setting %s gave score %f" % (cur_params, score))

      if score > cur_param_best_score:
        cur_param_best_score = score
        cur_best_param_val = param

      if score > best_total_score:
        best_total_score = score
        best_total_setting = list(cur_params)
        print("\nNEW BEST SCORE: %s with score %f\n" %
            (best_total_setting, best_total_score))

    cur_setting[params_idx] = cur_best_param_val
    print("Updated param %s to %s" %
        (param_names[params_idx], cur_best_param_val))
    print("Current param settings: %s for %s\n" % (cur_setting, param_names))

print("Best score was %f, obtained with %s" %
    (best_total_score, best_total_setting))

import os

import numpy as np
import tensorflow as tf

# source: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
def load_data(task_id, data_dir, vocab_file, tokenizer, batch_size):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled", 
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }
    babi_name = babi_map[task_id]
    babi_test_name = babi_map[task_id]
    train_q, train_a = _init_babi(os.path.join(data_dir, '%s_train.txt' % babi_name))
    test_q, test_a = _init_babi(os.path.join(data_dir, '%s_test.txt' % babi_test_name))

    # Create and return iterators.
    tf_vocab = tf.contrib.lookup.index_table_from_file(vocab_file,
        default_value=0)
    train_it = _get_iterator(train_q, train_a, tf_vocab, batch_size)
    test_it = _get_iterator(test_q, test_a, tf_vocab, len(test_a))
    return train_it, test_it

def _get_iterator(questions, answers, tf_vocab, batch_size):
    dataset_q = tf.data.Dataset.from_tensor_slices(questions)
    dataset_a = tf.data.Dataset.from_tensor_slices(answers)
    dataset = tf.data.Dataset.zip((dataset_q, dataset_a))

    # Shuffle the dataset randomly.
    dataset = dataset.shuffle(1000)

    # Split words in the sentence.
    dataset = dataset.map(lambda q, a: (tf.string_split([q]).values, a))

    # Convert words to word_ids.
    dataset = dataset.map(
        lambda q, a: (tf.cast(tf_vocab.lookup(q), tf.int32),
                      tf.cast(tf_vocab.lookup(a), tf.int32)))

    # Add question length.
    dataset = dataset.map(lambda q, a: (q, a, tf.size(q)))

    # The batching function batches entries and pads shorter questions
    # with sos symbols.
    eos_id = tf.cast(tf_vocab.lookup(tf.constant("</s>")), tf.int32)
    def batching_func(x):
      return x.padded_batch(
          batch_size,
          # Only the question is a variable length vector.
          padded_shapes=(tf.TensorShape([None]),  # question
                         tf.TensorShape([]),      # answer
                         tf.TensorShape([])),     # question length
          # Pad the extra values with an end-of-sentence token.
          padding_values=(eos_id,  # question
                          0,           # answer -- unused
                          0))          # question length -- unused
    dataset = batching_func(dataset)

    return dataset.make_initializable_iterator()

# Loads the vocabulary in a regular python dict.
def load_vocab(filename):
  w2i = {}
  i2w = {}
  with open(filename) as f:
    i = 0
    for line in f:
      word = line[:-1]
      if word in w2i:
        # word already in vocab
        continue
      w2i[word] = i
      i2w[i] = word
      if i == 0 and word != "<unk>":
        print("WARNING: first word in vocabulary file must be <unk>")
      i += 1
  return w2i, i2w

# adapted from: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
def _init_babi(fname):

    questions = []
    answers = []
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            C = ""
            Q = ""
            A = ""

        line = line.strip()
        line = line.replace('.', ' . ')
        line = line.replace('?', ' ? ')
        line = line[line.find(' ')+1:]
        if line.find('?') == -1:
            C += line.lower()
        else:
            idx = line.find('?') + 1
            tmp = line[idx+1:].split('\t')
            Q = C + line[:idx].lower()
            A = tmp[1].strip().lower()

            questions.append(tf.constant(Q))
            answers.append(tf.constant(A))
    return questions, answers

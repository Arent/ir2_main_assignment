import os

import numpy as np
import tensorflow as tf

# source: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
def load_data(id, test_id, data_dir, w2i, tokenizer, batch_size):
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
    if (test_id == ""):
        test_id = id 
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    babi_train_raw = _init_babi(os.path.join(data_dir, '%s_train.txt' % babi_name))
    babi_test_raw = _init_babi(os.path.join(data_dir, '%s_test.txt' % babi_test_name))

    return _process_data(babi_train_raw, tokenizer, w2i), _process_data(babi_test_raw, tokenizer, w2i)


def load_vocab(filename):
  w2i = {}
  i2w = {}
  with open(filename) as f:
    for i, line in enumerate(f):
      word = line[:-1]
      w2i[word] = i
      i2w[i] = word
  return w2i, i2w

def _to_index(word, w2i):
  return w2i[word] if word in w2i else w2i['<unk>']

# adapted from: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
def _init_babi(fname):

    tasks = []
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
            Q = C + line[:idx]
            A = tmp[1].strip()

            tasks.append((Q, A))
    return tasks

def _process_data(data, tokenizer, w2i):
    def to_index(word):
      return w2i[word] if word in w2i else w2i['<unk>']

    data = map(lambda x: (tokenizer.tokenize(x[0].lower()), x[1]), data)
    data = map(lambda x: ([to_index(w) for w in x[0]], to_index(x[1])), data)
    return data

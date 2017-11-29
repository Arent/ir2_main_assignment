import os

import numpy as np
import tensorflow as tf

# source: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
def load_data(task_id, data_dir, vocab_file, tokenizer, batch_size, val_split, prep=True):
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
        "1-10k": "../en-10k/qa1_single-supporting-fact",
        "2-10k": "../en-10k/qa2_two-supporting-facts",
        "3-10k": "../en-10k/qa3_three-supporting-facts",
        "4-10k": "../en-10k/qa4_two-arg-relations",
        "5-10k": "../en-10k/qa5_three-arg-relations",
        "6-10k": "../en-10k/qa6_yes-no-questions",
        "7-10k": "../en-10k/qa7_counting",
        "8-10k": "../en-10k/qa8_lists-sets",
        "9-10k": "../en-10k/qa9_simple-negation",
        "10-10k": "../en-10k/qa10_indefinite-knowledge",
        "11-10k": "../en-10k/qa11_basic-coreference",
        "12-10k": "../en-10k/qa12_conjunction",
        "13-10k": "../en-10k/qa13_compound-coreference",
        "14-10k": "../en-10k/qa14_time-reasoning",
        "15-10k": "../en-10k/qa15_basic-deduction",
        "16-10k": "../en-10k/qa16_basic-induction",
        "17-10k": "../en-10k/qa17_positional-reasoning",
        "18-10k": "../en-10k/qa18_size-reasoning",
        "19-10k": "../en-10k/qa19_path-finding",
        "20-10k": "../en-10k/qa20_agents-motivations",
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

    c, q, a = _init_babi(os.path.join(data_dir, '%s_train.txt' % babi_name), prep=prep)
    test_c, test_q, test_a = _init_babi(os.path.join(data_dir, '%s_test.txt' % babi_test_name), prep=prep)

    # Pick out some random data for validation.
    shuffled_indices = np.random.permutation(len(c))
    split_index = int(round(val_split * len(c)))
    train_indices = shuffled_indices[split_index:]
    val_indices = shuffled_indices[:split_index]
    train_c = [c[idx] for idx in train_indices]
    train_q = [q[idx] for idx in train_indices]
    train_a = [a[idx] for idx in train_indices]
    val_c = [c[idx] for idx in val_indices]
    val_q = [q[idx] for idx in val_indices]
    val_a = [a[idx] for idx in val_indices]

    # Return the training and test prepared for training.
    # Otherwise return the parsed data
    if prep:

        # Create and return iterators.
        tf_vocab = tf.contrib.lookup.index_table_from_file(vocab_file,
            default_value=0)
        train_it = _get_iterator(train_c, train_q, train_a, tf_vocab, batch_size)
        val_it = _get_iterator(val_c, val_q, val_a, tf_vocab, len(val_a))
        test_it = _get_iterator(test_c, test_q, test_a, tf_vocab, len(test_a))
        return train_it, val_it, test_it, tf_vocab
    else:
        train = zip(train_c, train_q, train_a)
        val = zip(val_c, val_q, val_a)
        test = zip(test_c, test_q, test_a)
        return train, val, test, None

def _get_iterator(contexts, questions, answers, tf_vocab, batch_size):
    dataset_c = tf.data.Dataset.from_tensor_slices(contexts)
    dataset_q = tf.data.Dataset.from_tensor_slices(questions)
    dataset_a = tf.data.Dataset.from_tensor_slices(answers)
    dataset = tf.data.Dataset.zip((dataset_c, dataset_q, dataset_a))

    # Shuffle the dataset randomly.
    dataset = dataset.shuffle(1000)

    # Split words in the sentence.
    dataset = dataset.map(lambda c, q, a: (tf.string_split([c]).values,
        tf.string_split([q]).values, tf.string_split([a]).values))

    # Convert words to word_ids.
    dataset = dataset.map(
        lambda c, q, a: (tf.cast(tf_vocab.lookup(c), tf.int32),
                      tf.cast(tf_vocab.lookup(q), tf.int32),
                      tf.cast(tf_vocab.lookup(a), tf.int32)))

    # Create an answer input prefixed with <s> and an answer output postfixed with </s>
    sos_id = tf.cast(tf_vocab.lookup(tf.constant("<s>")), tf.int32)
    eos_id = tf.cast(tf_vocab.lookup(tf.constant("</s>")), tf.int32)
    dataset = dataset.map(
        lambda c, q, a: (c,                            # context
                         q,                            # question
                         tf.concat(([sos_id], a), 0),  # answer_input
                         tf.concat((a, [eos_id]), 0))) # answer_output


    # Add context, question, and answer length.
    dataset = dataset.map(lambda c, q, answer_input, answer_output: (c, q,
        answer_input, answer_output, tf.size(c), tf.size(q),
        tf.size(answer_input)))

    # The batching function batches entries and pads shorter questions
    # with sos symbols.
    def batching_func(x):
      return x.padded_batch(
          batch_size,
          # Only the question is a variable length vector.
          padded_shapes=(tf.TensorShape([None]),  # context
                         tf.TensorShape([None]),  # question
                         tf.TensorShape([None]),  # answer_input
                         tf.TensorShape([None]),  # answer_output
                         tf.TensorShape([]),      # context length
                         tf.TensorShape([]),      # question length
                         tf.TensorShape([])),     # answer length
          # Pad the extra values with an end-of-sentence token.
          padding_values=(eos_id,      # context
                          eos_id,      # question
                          eos_id,      # answer_input
                          eos_id,      # answer_output
                          0,           # context length -- unused
                          0,           # question length -- unused
                          0))          # answer length -- unused
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
def _init_babi(fname, prep=True):

    contexts = []
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
            Q = line[:idx].lower()
            A = " ".join(tmp[1].strip().lower().split(","))

            if prep:
                questions.append(tf.constant(Q))
                answers.append(tf.constant(A))
                contexts.append(tf.constant(C))
            else:
                questions.append(Q)
                answers.append(A)
                contexts.append(C)
    return contexts, questions, answers

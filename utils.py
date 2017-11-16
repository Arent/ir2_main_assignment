import os
import random
import numpy as np
import tensorflow as tf

# source: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
def load_data(id, test_id, data_dir, w2i, tokenizer, batch_size, seperate_context=False):
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
    babi_train_raw = _init_babi(os.path.join(data_dir, '%s_train.txt' % babi_name), seperate_context)
    babi_test_raw = _init_babi(os.path.join(data_dir, '%s_test.txt' % babi_test_name), seperate_context)

    return _process_data(babi_train_raw, tokenizer, w2i), _process_data(babi_test_raw, tokenizer, w2i)


def load_vocab(filename):
  w2i = {}
  i2w = {}
  with open(filename) as f:
    lines = f.readlines()
    random.shuffle(lines)

    for i, line in enumerate(lines):
      word = line[:-1]
      w2i[word] = i
      i2w[i] = word
  return w2i, i2w

def _to_index(word, w2i):
  return w2i[word] if word in w2i else w2i['<unk>']

# adapted from: https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
def _init_babi_old(fname):

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


def _init_babi(fname, seperate_context):

    tasks = []
    for i, line in enumerate(open(fname)):
        #Find the first space in the line. The id is the characters left from that.
        id = int(line[0:line.find(' ')]) 
        #The id represent the line ids for one question. Reset if id =1
        if id == 1:
            C = "" #Context
            Q = "" #Question
            A = "" #Answer

        line = line.strip() 
        line = line.replace('.', ' . ')
        line = line.replace('?', ' ? ')
        line = line[line.find(' ')+1:]
        
        #Append the the lines to the context if the question hasn't arised yet
        if line.find('?') == -1:
            C += line.lower()
        else:
            #Question mark is found. Parse the answer and save the context, question and answer
            #Last line contains the question annser, the id's of the supporting facts and a tab

            idx = line.find('?') + 1
            Q = line[:idx]

            tmp = line[idx+1:].split('\t')
            A = tmp[1].strip()

            if seperate_context:
                tasks.append((C, Q, A))
            else:
                tasks.append((C + Q, A))
    return tasks


def _process_data(data_list, tokenizer, w2i):
    def to_index(word):
      return w2i[word] if word in w2i else w2i['<unk>']
    
    data_list_ids= [] 
    for data_tuple in data_list:
        data_tuple_words = [tokenizer.tokenize(element.lower()) for element in list(data_tuple)]
        new_tuple = [ [to_index(word) for word in element] for element in data_tuple_words]
        data_list_ids.append(new_tuple)

    return data_list_ids



def pad_to_k(x, k):    
    if len(x) < k:
        return(np.pad(x, (0, k-len(x)) , 'constant'))
    elif len(x) > k:
        return x[0:k]
    else:
        return x


def pad_results(data_list, k):
    lengths = np.array([len(example) for example in data_list])
    padded_examples = np.array([pad_to_k(example, k) for example in data_list ])

    return padded_examples.astype(np.int32), lengths.astype(np.int32)

def get_max_length(data_tuple_list):
    context, question, answer =  list(zip(*data_tuple_list))
    return max(len(l) for l in context + question)

def batch_generator(data, batch_size, max_length):
    '''
    This function:
    - unpacks and pads the data
    - Randomly shuffles the data
    - Returns a batch_size partion of the data each time it is called. 
    '''
    
    random.shuffle(data)
    size = len(data)
    #unpack data
    context, question, answer  = list(zip(*data))
    answer = np.array(answer).flatten()

   
    #Pad quesiton and context data to max length
    context, length_context = pad_results(context, max_length)

    question, length_question = pad_results(question, max_length)

    batch_partitions = np.linspace(start=0, stop=size - (size%batch_size), num=int(size/batch_size),endpoint=False)
    for idx in batch_partitions:
        start = int(idx)
        end = int(idx + batch_size)
        context_batch = (context[start:end, :], length_context[start:end])
        question_batch = (question[start:end, :], length_question[start:end])
        answer_batch = answer[start:end]

        yield context_batch, question_batch, answer_batch





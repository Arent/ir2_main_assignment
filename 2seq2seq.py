from nltk.tokenize.moses import MosesTokenizer

import data_utils 
import argparse

def write(data, path, task, eval_type):
    q_f = open(path + '/questions_' + str(task) + '_' + eval_type, 'w')
    a_f = open(path + '/answers_' + str(task) + '_' + eval_type, 'w')
    for c, q, a in data:
        print(c + q, file=q_f)
        print(a, file=a_f)


def main(args):
    tokenizer = MosesTokenizer()

    train, test = data_utils.load_data(args.task, args.data_dir, args.vocab, 
                                                   tokenizer, None, prep=False)


    write(train, args.target_dir, args.task, "train")
    write(test, args.target_dir, args.task, "test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/en",
                        help="Directory containing the data.")
    parser.add_argument("--target_dir", type=str, default="data/seq2seq",
                        help="Directory containing the data.")
    parser.add_argument("--vocab", type=str, default="data/vocab.txt",
                        help="Vocabulary file")
    parser.add_argument("--task", type=str, default="1",
                        help="Task number")
    parser.add_argument('--seperate_context', type=bool, default=False,
                        help='seperate the context from the question')
    
    args = parser.parse_args()
    main(args)
from nltk.tokenize.moses import MosesTokenizer
from collections import defaultdict

import operator
import glob
import os


filenames = glob.glob("en-10k/*.txt")

tokenizer = MosesTokenizer()
vocab = defaultdict(int)

for filename in filenames:
  with open(filename) as f:
    for line in f:
      tokens = tokenizer.tokenize(line.lower())
      for token in tokens:
        vocab[token] += 1

words_by_freq = sorted(vocab.items(), key=operator.itemgetter(1))[::-1]
with open("vocab.txt", "w+") as f:
  f.write("<unk>\n")
  f.write("<s>\n")
  f.write("</s>\n")

  for word, count in words_by_freq:
    if word != "<unk>" and word != "<s>" and word != "</s>":
      f.write(word)
      f.write("\n")

print("Wrote vocabulary to vocab.txt")

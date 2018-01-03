# IR2 QA Project on bAbi Toy Tasks
The main assignment of IR2. 

# Setup

## Requirements	

> import nltk
> nltk.download('perluniprops')
> nltk.download('nonbreaking_prefixes')

## Fetch all the datasets
Before running any code in the codebase, please run:

> ./setup.sh

or

> sh setup.sh

This will download the bAbI dataset and converts it to the format required by the seq2seq framework. 
Note: Creating the dataset requires an active internet connection and can take a few minutes.

## Testing the convolutional seq2seq implementation.
The following steps describe how to train or experiment with the conv seq2seq model.

# Documents

### bAbi
- [The paper introducing the bAbi toy tasks](./question-answering-2015.pdf)

- bAbi website: https://research.fb.com/downloads/babi/

## Related Work
### Papers

LSTM with Attention

- https://arxiv.org/pdf/1409.0473.pdf Paper that introduces attention in an encoder-decoder architecture for NMT, this is the same type of architecture that we will use. Known as Bahdanau's attention, implemented in Tensorflow already.
- https://arxiv.org/pdf/1508.04025.pdf Paper that proposes a lot of variations on Bahdanau's attention version above, one of which is used a lot today as an alternative to Bahdanau attention, known as Luong attention. Luong is the guy who wrote the tensorflow/nmt tutorial. Also implemented in Tensorflow already.
- https://arxiv.org/pdf/1509.06664.pdf

Convolutional Sequence to Sequence

- https://arxiv.org/pdf/1705.03122.pdf

Memory Networks

- https://arxiv.org/pdf/1503.08895.pdf Memory networks using weak supervision, linked to by the FB bAbi folks.

### Tutorials
On Deep learning NLP and memory networks

- http://web.stanford.edu/class/cs224n/syllabus.html Stanford NLP + Deep Learning course
- http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture16-DMN-QA.pdf / https://www.youtube.com/watch?v=T3octNTE7Is Lecture from that same course on dynamic memory networks.

LSTM tutorial + attention

- https://github.com/tensorflow/nmt a good tutorial on LSTM encoder-decoder systems with attention in Tensorflow.
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/ a good read on how LSTM cells work and the intuition behind them.
- https://distill.pub/2016/augmented-rnns/

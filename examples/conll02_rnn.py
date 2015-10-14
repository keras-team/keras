from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from pprint import pprint
import os
import itertools

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import recurrent
from keras.models import Sequential

'''
    Train a RNN on the ConLL 2002 Language-Independent Named Entity Recognition task.

    This is a very simple attempt of solving Language-Independent Named Entity
    Recognition task, first truncates sentences into ngram sub-sentences, after
    embedding the inputs, feeds them into a Recurrent Neural Network.

    Notes:
    For the resources related to the ConLL 2002 task, refer to:
    http://www.cnts.ua.ac.be/conll2002/ner/    

'''

def measure(predict, groundtruth, vocab_label_size, bgn_label_idx):
    '''
        get precision, recall, f1 score  
    '''
    tp = []
    fp = []
    fn = []
    recall = 0
    precision = 0
    for i in range(vocab_label_size):
        tp.append(0)
        fp.append(0)
        fn.append(0)

    for i in range(len(groundtruth)):
        if groundtruth[i] == predict[i]:
            tp[groundtruth[i]] += 1
        else:
            fp[predict[i]] += 1
            fn[groundtruth[i]] += 1

    for i in range(vocab_label_size):
        # do not count begin label
        if i == bgn_label_idx:
            continue
        if tp[i] + fp[i] == 0:
            precision += 1
        else:
            precision += float(tp[i]) / float(tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall += 1
        else:
            recall += float(tp[i]) / float(tp[i] + fn[i])

    precision /= (vocab_label_size - 1)
    recall /= (vocab_label_size - 1)
    pprint(tp)
    pprint(fp)
    pprint(fn)
    f1 = 2 * float(precision) * float(recall) / (precision + recall)
    print ('precision: %f, recall: %f, f1 score on testa is %f' % (precision, recall, f1))


def get_file(url):
    '''
        download and unzip dataset
    '''
    fname = url.split('/')[-1]
    if not os.path.exists(fname):
        os.system('curl -O ' + url)

    dirname = os.popen("tar -tzf " + fname + " | sed -e 's@/.*@@' | uniq").read().strip()
    if not os.path.exists(dirname):
        os.system('tar -xvzf ' + fname)
    dirname = dirname + '/data/'
    tmp = dirname + 'esp.testa'
    if not os.path.exists(tmp):
        os.system('gunzip ' + tmp)
    tmp = dirname + 'esp.testb'
    if not os.path.exists(tmp):
        os.system('gunzip ' + tmp)
    tmp = dirname + 'esp.train'
    if not os.path.exists(tmp):
        os.system('gunzip ' + tmp)

def add_begin_end_word(sentences, labels, ngram):
    '''
        adds begin and end words for each sentence
        for bi-directional rnn, we need end words
        '__BGN__' for begin
        '__END__' for end
    '''
    BEGIN_WORD = '__BGN__'
    BEGIN_LABEL = '__BGN__'

    for i in range(len(sentences)):
        sentences[i] = [BEGIN_WORD] * (ngram - 1) + sentences[i]
        labels[i] = [BEGIN_LABEL] * (ngram - 1) + labels[i]


def load_ner_data(fname):
    '''
        load ner data into arrays
    '''
    sentences = []
    labels = []
    with open(fname) as fin:
        sentence = []
        label = []
        for l in fin.xreadlines():
            l = l.strip()
            if l == '': # sentence ends
                sentences.append(sentence)
                labels.append(label)

                sentence = []
                label = []
            else:
                w, l = l.split(' ')
                sentence.append(w)
                label.append(l)

        if len(sentence) != 0:
            sentences.append(sentence)
            labels.append(label)

    return sentences, labels


def to_ngrams(sentences, labels, ngram):
    '''
        truncates sentences into sub-sentences
    '''
    ngrams = []
    ngram_labels = []

    for s, l in zip(sentences, labels):
        for i in range(len(s) - ngram + 1):
            ngrams.append(s[i : i + ngram])
            ngram_labels.append(l[i + ngram - 1])

    return ngrams, ngram_labels

def to_vocabulary(lst_of_lst):
    # use from_iterable to flat the lst
    return set(itertools.chain.from_iterable(lst_of_lst))


def to_vocab_index(vocab_index_dict, ngrams):
    ngrams_index = []
    for ngram in ngrams:
        ngrams_index.append([vocab_index_dict[w] for w in ngram])
    return ngrams_index


def to_label_index(label_index_dict, labels):
    return [label_index_dict[l] for l in labels]


def get_data(path, ngram, begin_end_word = True):
    '''
        converts conll 2002 dataset into Keras support type
    '''
    # read data into arrays (array of arrays)
    train_sentences, train_labels = load_ner_data(path + '/esp.train')
    testa_sentences, testa_labels = load_ner_data(path + '/esp.testa')
    testb_sentences, testb_labels = load_ner_data(path + '/esp.testb')

    # add begin and end words into each sentences (the amount of b&e words depends on ngram)
    #   e.g. when ngram == 3
    #   This is an example ---> __BGN__ __BGN__ This is an example __END__ __END__ 
    if begin_end_word:
        add_begin_end_word(train_sentences, train_labels, ngram)
        add_begin_end_word(testa_sentences, testa_labels, ngram)
        add_begin_end_word(testb_sentences, testb_labels, ngram)

    # truncate sentences into ngram sub-sentences, and store the words' number in dictionary into train_x/test_x
    #   e.g. when ngram == 3
    #   __BGN__ __BGN__ This is an example __END__ __END__ 
    #   become
    #   #__BGN__ #__BGN__ #This
    #   #__BGN__ #This #is
    #   #This #is #an
    #   #is #an #example
    #   #an #example #__END__
    #   #example #__END__ #__END__ 
    #   where '#' represents a word's number in dictionary
    train_x, train_y = to_ngrams(train_sentences, train_labels, ngram)
    testa_x, testa_y = to_ngrams(testa_sentences, testa_labels, ngram)
    testb_x, testb_y = to_ngrams(testb_sentences, testb_labels, ngram)

    # generate vocabulary for x and y
    vocab = to_vocabulary(train_sentences) | to_vocabulary(testa_sentences) | to_vocabulary(testb_sentences)
    vocab_label = to_vocabulary(train_labels)

    # generate a dictionary mapped from word / label to its index
    vocab_index_dict = {w: i for i, w in enumerate(vocab)}
    label_index_dict = {l: i for i, l in enumerate(vocab_label)}

    # convert from words to words' indexes
    train_x = to_vocab_index(vocab_index_dict, train_x)
    testa_x = to_vocab_index(vocab_index_dict, testa_x)
    testb_x = to_vocab_index(vocab_index_dict, testb_x)

    # convert from labels to labels' indexes
    train_y = to_label_index(label_index_dict, train_y)
    testa_y = to_label_index(label_index_dict, testa_y)
    testb_y = to_label_index(label_index_dict, testb_y)

    bgn_label_idx = label_index_dict['__BGN__']

    return train_x, train_y, testa_x, testb_x, testa_y, testb_y, vocab, vocab_label, bgn_label_idx


rnn = recurrent.GRU
embedding_size = 256
hidden_size = 256
batch_size = 128
epochs = 20
ngram = 3

# download CoNLL 2002 dataset from official website and unzip if local file doesn't exist
get_file('http://www.cnts.ua.ac.be/conll2002/ner.tgz')

train_x, train_y, testa_x, testb_x, testa_y, testb_y, vocab, vocab_label, bgn_label_idx = get_data('./ner/data', ngram)
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
vocab_label_size = len(vocab_label)
print('vocab_size = ', vocab_size)
print('vocab_label_size =', vocab_label_size)

# pad x
print('Padding train data...')
train_x_pad = sequence.pad_sequences(train_x, maxlen=ngram)
print('Padding test data a...')
testa_x_pad = sequence.pad_sequences(testa_x, maxlen=ngram)
print('Padding test data b...')
testb_x_pad = sequence.pad_sequences(testb_x, maxlen=ngram)

print('train_x.shape = {}'.format(train_x_pad.shape))
print('testa_x.shape = {}'.format(testa_x_pad.shape))
print('testb_x.shape = {}'.format(testb_x_pad.shape))

# convert class vectors to binary class matrices
train_y_categorical = np_utils.to_categorical(train_y, vocab_label_size)
testa_y_categorical = np_utils.to_categorical(testa_y, vocab_label_size)
testb_y_categorical = np_utils.to_categorical(testb_y, vocab_label_size)

print('train_y_categorical.shape = {}'.format(train_y_categorical.shape))
print('testa_y_categorical.shape = {}'.format(testa_y_categorical.shape))
print('testb_y_categorical.shape = {}'.format(testb_y_categorical.shape))

# build model
print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, mask_zero=True))
model.add(rnn(hidden_size, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(vocab_label_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')

# train & test model
model.fit(train_x_pad, train_y_categorical, batch_size=batch_size, nb_epoch=epochs, show_accuracy=True, verbose=1)
testa_y_predict = model.predict_classes(testa_x_pad, batch_size=batch_size,verbose=1)
testb_y_predict = model.predict_classes(testb_x_pad, batch_size=batch_size,verbose=1)

# get precision, recall, f1 score  
print('tesing test data a...')
measure(testa_y_predict, testa_y, vocab_label_size, bgn_label_idx)
print('tesing test data b...')
measure(testb_y_predict, testb_y, vocab_label_size, bgn_label_idx)

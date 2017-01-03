'''Train CRF and BiLSTM-CRF on CONLL2000 chunking data, similar to https://arxiv.org/pdf/1508.01991v1.pdf.

For CRF, we get ~0.93 after 10 epochs
For BiLSTM-CRF, we get ~0.94 after 3 epochs
'''

from __future__ import print_function
import numpy
from nltk.corpus import conll2000
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers import CRF
from collections import Counter
from sklearn.metrics import classification_report

numpy.random.seed(1223)

if not hasattr(conll2000, 'tagged_sents'):
    import nltk
    nltk.download('conll2000')

tagged_sents = [s for s in conll2000.tagged_sents()]
class_labels = sorted(list(set(w[1] for s in tagged_sents for w in s)))

N = len(tagged_sents)
train_nb = N * 0.8
idx = numpy.random.choice(numpy.arange(N), N, replace=False)
train = [tagged_sents[i] for i in idx[:train_nb]]
test = [tagged_sents[i] for i in idx[train_nb:]]

word_counts = Counter(w[0].lower() for s in train for w in s)
vocab = ['<pad>', '<unk>'] + [w for w, f in word_counts.iteritems() if f >= 3]
word2idx = dict((w, i) for i, w in enumerate(vocab))

def process_data(data, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    x = [[w[0].lower() for w in s] for s in data]
    y = [[w[1] for w in s] for s in data]
    x = pad_sequences([[word2idx.get(w[0].lower(), 1) for w in s] for s in data], maxlen=maxlen)
    y = pad_sequences([[class_labels.index(w[1])for w in s] for s in data], maxlen=maxlen)
    if onehot:
        y = numpy.eye(len(class_labels), dtype='float32')[y]
        return x, y
    else:
        return x, numpy.expand_dims(y, 2)

train_x, train_y = process_data(train)
test_x, test_y = process_data(test)

# --------------
# 1. Regular CRF
# --------------

print('==== training CRF ====')

model = Sequential()
model.add(Embedding(len(vocab), 200, mask_zero=True)) # Random embedding
crf = CRF(len(class_labels), sparse_target=True)
model.add(crf)
model.summary()

model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
model.fit(train_x, train_y, nb_epoch=10, validation_data=[test_x, test_y])

test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
test_y_true = test_y[test_x > 0]

print('\n---- Result of CRF ----\n')
print(classification_report(test_y_true, test_y_pred, target_names=class_labels))

# -------------
# 2. BiLSTM-CRF
# -------------

print('==== training BiLSTM-CRF ====')

model = Sequential()
model.add(Embedding(len(vocab), 200, mask_zero=True)) # Random embedding
model.add(Bidirectional(LSTM(100, return_sequences=True)))
crf = CRF(len(class_labels), sparse_target=True)
model.add(crf)
model.summary()

model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
model.fit(train_x, train_y, nb_epoch=10, validation_data=[test_x, test_y])

test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
test_y_true = test_y[test_x > 0]

print('\n---- Result of BiLSTM-CRF ----\n')
print(classification_report(test_y_true, test_y_pred, target_names=class_labels))

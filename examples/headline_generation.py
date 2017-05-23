"""Trains an recurrent net to generate headlines given article body text. 

Inspired by http://nlp.stanford.edu/courses/cs224n/2015/reports/1.pdf.

This uses part of the twenty news groups data set, which is publicly available at: 
    
http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz
"""

import numpy as np
np.random.seed(427)

import sys
sys.setrecursionlimit(10000) # For the addition of dropout in the GRU layers. 
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers.recurrent import GRU
from keras.layers.core import Dense
from keras.utils.np_utils import to_categorical
from sklearn.datasets import fetch_20newsgroups
import re

# Sci.crypt has the most articles, and a high average headline word count. 
articles = fetch_20newsgroups(subset='all', categories=['sci.crypt'], shuffle=False).data

# Each article is one long string, and the parts to grab are stored within certain
# fields ("Subject:" for the headline, "Lines:" for the body). 
def get_bodies_headlines(articles): 
    bodies, headlines = [], []
    
    for article in articles:
        # Grab all article text after the field name up to the next field name. 
        headline = re.findall("(?<=Subject:)(.*)", article)
        body = re.findall("(?<=Lines:)(?s)(.*)", article)
 
        if body and headline: 
            bodies.append(body[0])
            headlines.append(headline[0])

    return bodies, headlines

bodies, headlines = get_bodies_headlines(articles)

maxlen = 50 # Maximum length of X sequences to feed into the model. 
embedding_dim = 50 # Number of dimensions to have in the word vectors. 
 
# Lower and remove punctuation.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(bodies)
tokenizer.fit_on_texts(headlines)

# Use for encoding/decoding words. 
word_idx_dct = tokenizer.word_index
word_idx_dct['\n'] = 0 # End of Sequence (EOS) character. 
idx_word_dct = {idx: word for word, idx in word_idx_dct.items()}
idx_word_dct[0] = '\n' # EOS character. 
vocab_size = len(word_idx_dct)

# Generate vectorized versions of the texts (list of ints). 
bodies = list(tokenizer.texts_to_sequences_generator(bodies))
headlines = list(tokenizer.texts_to_sequences_generator(headlines))

def gen_Xy_pairs(bodies, headlines, maxlen): 
    """Prep inputs into fixed-length X, y pairs. 
    
    Create X, y pairs for each body/headline pair. For the first ob. from 
    each body/headline pair, take the first `maxlen` words of the body followed by an
    EOS (0 in the `idx_word_dct`) for X and the first word of the headline for y.
    Generate subsequent X's by stepping through the previous X by one word, and
    tacking the previous y onto the end. Generate subsequent y's by stepping through
    the headline, stopping when the end of the headline is reached. 
    """

    Xs, ys = [], []
    for body, headline in zip(bodies, headlines): 
        # Add the EOS into the headline. 
        headline.append(0)
        if len(body) > len(headline) + maxlen: 
            for idx, word in enumerate(headline): 
                # Grab the words we want from the body, append the EOS, and tack on 
                # any words of the headline that should be added. 
                X = body[idx:maxlen] + [0] + headline[:idx]
                y = headline[idx] 
                Xs.append(X)
                ys.append(y)

    return Xs, ys 

# Split into train and test now so that the full body and headlines are still
# accessible later. Use 70% of the data for training, and 70% for test. 
split_point = int(len(bodies) * 0.50)
train_bodies, train_headlines = bodies[:split_point], headlines[:split_point]
test_bodies, test_headlines = bodies[split_point:], headlines[split_point:]

# Some of the headlines are different by only one or two words, since some articles
# are reponses to others. Filter out the body/headline pairs in the test set that are
# extremly simliar to anything in the training set. 
test_bodies_filtered, test_headlines_filtered = [], []
for test_body, test_hline in zip(test_bodies, test_headlines): 
    # Ignore the order and frequency of words. 
    test_hline_set = set(test_hline)
    remove = False
    for train_hline in train_headlines: 
        train_hline_set = set(train_hline)
        in_common = len(test_hline_set.intersection(train_hline_set))
        pct_in_common = in_common / len(test_hline_set)
        if pct_in_common >= 0.90: 
            remove = True
    if not remove: 
        test_bodies_filtered.append(test_body)
        test_headlines_filtered.append(test_hline)

test_bodies, test_headlines = test_bodies_filtered, test_headlines_filtered 
Xs_train, ys_train = gen_Xy_pairs(train_bodies, train_headlines, maxlen)
Xs_test, ys_test = gen_Xy_pairs(test_bodies, test_headlines, maxlen)

# The X pairs will simply keep the index corresponding to their place in the 
# word_idx_dct, while the y will become one-hot-encoded. This way, it'll line up
# with the output of the final `Dense` layer in the model below. 
X_train = np.array(Xs_train, dtype='int32') 
X_test = np.array(Xs_test, dtype='int32') 
y_train = to_categorical(ys_train, vocab_size)
y_test = to_categorical(ys_test, vocab_size)

dropout = 0
bodies_input = Input(shape=(maxlen + 1,), dtype='int32')
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                      dropout=dropout)(bodies_input)
layer = GRU(128, return_sequences=True, dropout_W=dropout,       
            dropout_U=dropout)(embedding)
layer = GRU(128, return_sequences=False, dropout_W=dropout, dropout_U=dropout)(layer)
layer = Dense(vocab_size, activation='softmax')(layer)

lstm_model = Model(input=bodies_input, output=layer)
lstm_model.compile(loss='categorical_crossentropy', optimizer='adagrad')
lstm_model.fit(X_train, y_train, batch_size=32, nb_epoch=10, 
               validation_data=(X_test, y_test))

def predict_sequence(lstm_model, x):
    """Predict an entire headline from the inputted x. 

    The inputted x will only contain words from the body (e.g. it'll be the first 
    X,y pair created for each body/headline pair in gen_`Xy`_pairs). Predict on it
    once to obtain the first word of the headline. Iteratively build up a sequence 
    by tacking on the last prediction and predicting again, until predicting the EOS
    character (or the sequence is longer than 10 words). 
    """

    # Cast to list so that we can append to it later. 
    y_pred = []
    
    # Keep going while getting non-zero predictions (zero is EOS), or until 
    # hitting a length of 10.
    pred = 10
    while pred and len(y_pred) < 10: 
        x_arr = np.array(x)[np.newaxis]
        pred_vector = lstm_model.predict(x_arr)
        pred = np.argmax(pred_vector)

        y_pred.append(pred)
        # Drop the first word to tack on the predicted y for the next 
        # prediction. 
        x = x[1:]
        x.append(pred)
    
    y_pred = np.array(y_pred)

    return y_pred

# Predict a sequence for the first X in each of the (X, y) pairs created for a given
# body/headline, and for the first 5 body/headlines. The first X in all of those 
# (X, y) pairs is the only one that will contain only words from the body.  
row_idx = 0
for hline in test_headlines[0:5]: 
    for word_num, word in enumerate(hline): 
        x = Xs_test[row_idx]
        # Generate a sequence only if y corresponds to the first word of the healine,
        # meaning that x only contains words from the body. 
        if not word_num: 
            y_pred = predict_sequence(lstm_model, x)
        
        row_idx += 1

    predicted_heading = ' '.join(idx_word_dct[idx] for idx in y_pred)
    actual_heading = ' '.join(idx_word_dct[idx] for idx in hline)

    print("---------------Actual-----------------")
    print(actual_heading)
    print("--------------Predicted---------------")
    print(predicted_heading)

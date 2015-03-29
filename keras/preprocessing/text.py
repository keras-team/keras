# -*- coding: utf-8 -*-
'''
    These preprocessing utils would greatly benefit
    from a fast Cython rewrite.
'''

import string
import numpy as np

def base_filter():
    f = string.punctuation
    f += '\t\n'
    return f

def text_to_word_sequence(text, filters=base_filter(), lower=True, split=" "):
    '''prune: sequence of characters to filter out
    '''
    if lower:
        text = text.lower()
    text = text.translate(string.maketrans("",""), filters)
    return text.split(split)


def one_hot(text, n):
    seq = text_to_word_sequence(text)
    return [abs(hash(w))%n for w in seq]


class Tokenizer(object):
    def __init__(self, filters=base_filter(), lower=True, nb_words=None):
        self.word_counts = {}
        self.word_docs = {}
        self.filters = filters
        self.lower = lower
        self.nb_words = nb_words
        self.document_count = 0

    def fit_on_texts(self, texts):
        '''
            required before using texts_to_sequences or texts_to_matrix
        '''
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1
        self.document_count = len(texts)

        wcounts = self.word_counts.items()
        wcounts.sort(key = lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        self.word_index = dict(zip(sorted_voc, range(len(sorted_voc))))

        self.index_docs = {}
        for w, c in self.word_docs.items():
            self.index_docs[self.word_index[w]] = c


    def fit_on_sequences(self, sequences):
        '''
            required before using sequences_to_matrix 
            (if fit_on_texts was never called)
        '''
        self.document_count = len(sequences)
        self.index_docs = {}
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                if i not in self.index_docs:
                    self.index_docs[i] = 1
                else:
                    self.index_docs[i] += 1


    def texts_to_sequences(self, texts):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words know by the tokenizer will be taken into account.
        '''
        nb_words = self.nb_words
        res = []
        for text in texts:
            seq = text_to_word_sequence(text, self.filters, self.lower)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        pass
                    else:
                        vect.append(i)
            res.append(vect)
        return res

    def texts_to_matrix(self, texts, mode="binary"):
        '''
            modes: binary, count, tfidf, freq
        '''
        sequences = self.texts_to_sequences(texts)
        return self.sequences_to_matrix(sequences, mode=mode)

    def sequences_to_matrix(self, sequences, mode="binary"):
        '''
            modes: binary, count, tfidf, freq
        '''
        if not self.nb_words:
            if self.word_index:
                nb_words = len(self.word_index)
            else:
                raise Exception("Specify a dimension (nb_words argument), or fit on some text data first")
        else:
            nb_words = self.nb_words

        if mode == "tfidf" and not self.document_count:
            raise Exception("Fit the Tokenizer on some data before using tfidf mode")

        X = np.zeros((len(sequences), nb_words))
        for i, seq in enumerate(sequences):
            if not seq:
                pass
            counts = {}
            for j in seq:
                if j >= nb_words:
                    pass
                if j not in counts:
                    counts[j] = 1.
                else:
                    counts[j] += 1
            for j, c in counts.items():
                if mode == "count":
                    X[i][j] = c
                elif mode == "freq":
                    X[i][j] = c/len(seq)
                elif mode == "binary":
                    X[i][j] = 1
                elif mode == "tfidf":
                    tf = np.log(c/len(seq))
                    df = (1 + np.log(1 + self.index_docs.get(j, 0)/(1 + self.document_count)))
                    X[i][j] = tf / df
                else:
                    raise Exception("Unknown vectorization mode: " + str(mode))
        return X



                



    

"""Utilities for text input preprocessing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_preprocessing import text

text_to_word_sequence = text.text_to_word_sequence
one_hot = text.one_hot
hashing_trick = text.hashing_trick
Tokenizer = text.Tokenizer
tokenizer_from_json = text.tokenizer_from_json

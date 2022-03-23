# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for text input preprocessing.

Deprecated: `tf.keras.preprocessing.text` APIs are not recommended for new code.
Prefer `tf.keras.utils.text_dataset_from_directory` and
`tf.keras.layers.TextVectorization` which provide a more efficient approach
for preprocessing text input. For an introduction to these APIs, see
the [text loading tutorial]
(https://www.tensorflow.org/tutorials/load_data/text)
and [preprocessing layer guide]
(https://www.tensorflow.org/guide/keras/preprocessing_layers).
"""
# pylint: disable=invalid-name
# pylint: disable=g-classes-have-attributes
# pylint: disable=g-direct-tensorflow-import

import collections
import hashlib
import json
import warnings

import numpy as np
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.preprocessing.text.text_to_word_sequence')
def text_to_word_sequence(input_text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=' '):
  r"""Converts a text to a sequence of words (or tokens).

  Deprecated: `tf.keras.preprocessing.text.text_to_word_sequence` does not
  operate on tensors and is not recommended for new code. Prefer
  `tf.strings.regex_replace` and `tf.strings.split` which provide equivalent
  functionality and accept `tf.Tensor` input. For an overview of text handling
  in Tensorflow, see the [text loading tutorial]
  (https://www.tensorflow.org/tutorials/load_data/text).

  This function transforms a string of text into a list of words
  while ignoring `filters` which include punctuations by default.

  >>> sample_text = 'This is a sample sentence.'
  >>> tf.keras.preprocessing.text.text_to_word_sequence(sample_text)
  ['this', 'is', 'a', 'sample', 'sentence']

  Args:
      input_text: Input text (string).
      filters: list (or concatenation) of characters to filter out, such as
          punctuation. Default: ``'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n'``,
            includes basic punctuation, tabs, and newlines.
      lower: boolean. Whether to convert the input to lowercase.
      split: str. Separator for word splitting.

  Returns:
      A list of words (or tokens).
  """
  if lower:
    input_text = input_text.lower()

  translate_dict = {c: split for c in filters}
  translate_map = str.maketrans(translate_dict)
  input_text = input_text.translate(translate_map)

  seq = input_text.split(split)
  return [i for i in seq if i]


@keras_export('keras.preprocessing.text.one_hot')
def one_hot(input_text,
            n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' ',
            analyzer=None):
  r"""One-hot encodes a text into a list of word indexes of size `n`.

  Deprecated: `tf.keras.text.preprocessing.one_hot` does not operate on tensors
  and is not recommended for new code. Prefer `tf.keras.layers.Hashing` with
  `output_mode='one_hot'` which provides equivalent functionality through a
  layer which accepts `tf.Tensor` input. See the [preprocessing layer guide]
  (https://www.tensorflow.org/guide/keras/preprocessing_layers)
  for an overview of preprocessing layers.

  This function receives as input a string of text and returns a
  list of encoded integers each corresponding to a word (or token)
  in the given input string.

  Args:
      input_text: Input text (string).
      n: int. Size of vocabulary.
      filters: list (or concatenation) of characters to filter out, such as
        punctuation. Default:
        ```
        '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n
        ```,
        includes basic punctuation, tabs, and newlines.
      lower: boolean. Whether to set the text to lowercase.
      split: str. Separator for word splitting.
      analyzer: function. Custom analyzer to split the text

  Returns:
      List of integers in `[1, n]`. Each integer encodes a word
      (unicity non-guaranteed).
  """
  return hashing_trick(
      input_text,
      n,
      hash_function=hash,
      filters=filters,
      lower=lower,
      split=split,
      analyzer=analyzer)


@keras_export('keras.preprocessing.text.hashing_trick')
def hashing_trick(text,
                  n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' ',
                  analyzer=None):
  r"""Converts a text to a sequence of indexes in a fixed-size hashing space.

  Deprecated: `tf.keras.text.preprocessing.hashing_trick` does not operate on
  tensors and is not recommended for new code. Prefer `tf.keras.layers.Hashing`
  which provides equivalent functionality through a layer which accepts
  `tf.Tensor` input. See the [preprocessing layer guide]
  (https://www.tensorflow.org/guide/keras/preprocessing_layers)
  for an overview of preprocessing layers.

  Args:
      text: Input text (string).
      n: Dimension of the hashing space.
      hash_function: defaults to python `hash` function, can be 'md5' or
          any function that takes in input a string and returns a int.
          Note that 'hash' is not a stable hashing function, so
          it is not consistent across different runs, while 'md5'
          is a stable hashing function.
      filters: list (or concatenation) of characters to filter out, such as
          punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n``,
          includes basic punctuation, tabs, and newlines.
      lower: boolean. Whether to set the text to lowercase.
      split: str. Separator for word splitting.
      analyzer: function. Custom analyzer to split the text

  Returns:
      A list of integer word indices (unicity non-guaranteed).
      `0` is a reserved index that won't be assigned to any word.
      Two or more words may be assigned to the same index, due to possible
      collisions by the hashing function.
      The [probability](
          https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
      of a collision is in relation to the dimension of the hashing space and
      the number of distinct objects.
  """
  if hash_function is None:
    hash_function = hash
  elif hash_function == 'md5':
    hash_function = lambda w: int(hashlib.md5(w.encode()).hexdigest(), 16)

  if analyzer is None:
    seq = text_to_word_sequence(text, filters=filters, lower=lower, split=split)
  else:
    seq = analyzer(text)

  return [(hash_function(w) % (n - 1) + 1) for w in seq]


@keras_export('keras.preprocessing.text.Tokenizer')
class Tokenizer(object):
  """Text tokenization utility class.

  Deprecated: `tf.keras.preprocessing.text.Tokenizer` does not operate on
  tensors and is not recommended for new code. Prefer
  `tf.keras.layers.TextVectorization` which provides equivalent functionality
  through a layer which accepts `tf.Tensor` input. See the
  [text loading tutorial](https://www.tensorflow.org/tutorials/load_data/text)
  for an overview of the layer and text handling in tensorflow.

  This class allows to vectorize a text corpus, by turning each
  text into either a sequence of integers (each integer being the index
  of a token in a dictionary) or into a vector where the coefficient
  for each token could be binary, based on word count, based on tf-idf...

  By default, all punctuation is removed, turning the texts into
  space-separated sequences of words
  (words maybe include the `'` character). These sequences are then
  split into lists of tokens. They will then be indexed or vectorized.

  `0` is a reserved index that won't be assigned to any word.

  Args:
      num_words: the maximum number of words to keep, based
          on word frequency. Only the most common `num_words-1` words will
          be kept.
      filters: a string where each element is a character that will be
          filtered from the texts. The default is all punctuation, plus
          tabs and line breaks, minus the `'` character.
      lower: boolean. Whether to convert the texts to lowercase.
      split: str. Separator for word splitting.
      char_level: if True, every character will be treated as a token.
      oov_token: if given, it will be added to word_index and used to
          replace out-of-vocabulary words during text_to_sequence calls
      analyzer: function. Custom analyzer to split the text.
          The default analyzer is text_to_word_sequence
  """

  def __init__(self,
               num_words=None,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=' ',
               char_level=False,
               oov_token=None,
               analyzer=None,
               **kwargs):
    # Legacy support
    if 'nb_words' in kwargs:
      warnings.warn('The `nb_words` argument in `Tokenizer` '
                    'has been renamed `num_words`.')
      num_words = kwargs.pop('nb_words')
    document_count = kwargs.pop('document_count', 0)
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    self.word_counts = collections.OrderedDict()
    self.word_docs = collections.defaultdict(int)
    self.filters = filters
    self.split = split
    self.lower = lower
    self.num_words = num_words
    self.document_count = document_count
    self.char_level = char_level
    self.oov_token = oov_token
    self.index_docs = collections.defaultdict(int)
    self.word_index = {}
    self.index_word = {}
    self.analyzer = analyzer

  def fit_on_texts(self, texts):
    """Updates internal vocabulary based on a list of texts.

    In the case where texts contains lists,
    we assume each entry of the lists to be a token.

    Required before using `texts_to_sequences` or `texts_to_matrix`.

    Args:
        texts: can be a list of strings,
            a generator of strings (for memory-efficiency),
            or a list of list of strings.
    """
    for text in texts:
      self.document_count += 1
      if self.char_level or isinstance(text, list):
        if self.lower:
          if isinstance(text, list):
            text = [text_elem.lower() for text_elem in text]
          else:
            text = text.lower()
        seq = text
      else:
        if self.analyzer is None:
          seq = text_to_word_sequence(
              text, filters=self.filters, lower=self.lower, split=self.split)
        else:
          seq = self.analyzer(text)
      for w in seq:
        if w in self.word_counts:
          self.word_counts[w] += 1
        else:
          self.word_counts[w] = 1
      for w in set(seq):
        # In how many documents each word occurs
        self.word_docs[w] += 1

    wcounts = list(self.word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    # forcing the oov_token to index 1 if it exists
    if self.oov_token is None:
      sorted_voc = []
    else:
      sorted_voc = [self.oov_token]
    sorted_voc.extend(wc[0] for wc in wcounts)

    # note that index 0 is reserved, never assigned to an existing word
    self.word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

    self.index_word = {c: w for w, c in self.word_index.items()}

    for w, c in list(self.word_docs.items()):
      self.index_docs[self.word_index[w]] = c

  def fit_on_sequences(self, sequences):
    """Updates internal vocabulary based on a list of sequences.

    Required before using `sequences_to_matrix`
    (if `fit_on_texts` was never called).

    Args:
        sequences: A list of sequence.
            A "sequence" is a list of integer word indices.
    """
    self.document_count += len(sequences)
    for seq in sequences:
      seq = set(seq)
      for i in seq:
        self.index_docs[i] += 1

  def texts_to_sequences(self, texts):
    """Transforms each text in texts to a sequence of integers.

    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.

    Args:
        texts: A list of texts (strings).

    Returns:
        A list of sequences.
    """
    return list(self.texts_to_sequences_generator(texts))

  def texts_to_sequences_generator(self, texts):
    """Transforms each text in `texts` to a sequence of integers.

    Each item in texts can also be a list,
    in which case we assume each item of that list to be a token.

    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.

    Args:
        texts: A list of texts (strings).

    Yields:
        Yields individual sequences.
    """
    num_words = self.num_words
    oov_token_index = self.word_index.get(self.oov_token)
    for text in texts:
      if self.char_level or isinstance(text, list):
        if self.lower:
          if isinstance(text, list):
            text = [text_elem.lower() for text_elem in text]
          else:
            text = text.lower()
        seq = text
      else:
        if self.analyzer is None:
          seq = text_to_word_sequence(
              text, filters=self.filters, lower=self.lower, split=self.split)
        else:
          seq = self.analyzer(text)
      vect = []
      for w in seq:
        i = self.word_index.get(w)
        if i is not None:
          if num_words and i >= num_words:
            if oov_token_index is not None:
              vect.append(oov_token_index)
          else:
            vect.append(i)
        elif self.oov_token is not None:
          vect.append(oov_token_index)
      yield vect

  def sequences_to_texts(self, sequences):
    """Transforms each sequence into a list of text.

    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.

    Args:
        sequences: A list of sequences (list of integers).

    Returns:
        A list of texts (strings)
    """
    return list(self.sequences_to_texts_generator(sequences))

  def sequences_to_texts_generator(self, sequences):
    """Transforms each sequence in `sequences` to a list of texts(strings).

    Each sequence has to a list of integers.
    In other words, sequences should be a list of sequences

    Only top `num_words-1` most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.

    Args:
        sequences: A list of sequences.

    Yields:
        Yields individual texts.
    """
    num_words = self.num_words
    oov_token_index = self.word_index.get(self.oov_token)
    for seq in sequences:
      vect = []
      for num in seq:
        word = self.index_word.get(num)
        if word is not None:
          if num_words and num >= num_words:
            if oov_token_index is not None:
              vect.append(self.index_word[oov_token_index])
          else:
            vect.append(word)
        elif self.oov_token is not None:
          vect.append(self.index_word[oov_token_index])
      vect = ' '.join(vect)
      yield vect

  def texts_to_matrix(self, texts, mode='binary'):
    """Convert a list of texts to a Numpy matrix.

    Args:
        texts: list of strings.
        mode: one of "binary", "count", "tfidf", "freq".

    Returns:
        A Numpy matrix.
    """
    sequences = self.texts_to_sequences(texts)
    return self.sequences_to_matrix(sequences, mode=mode)

  def sequences_to_matrix(self, sequences, mode='binary'):
    """Converts a list of sequences into a Numpy matrix.

    Args:
        sequences: list of sequences
            (a sequence is a list of integer word indices).
        mode: one of "binary", "count", "tfidf", "freq"

    Returns:
        A Numpy matrix.

    Raises:
        ValueError: In case of invalid `mode` argument,
            or if the Tokenizer requires to be fit to sample data.
    """
    if not self.num_words:
      if self.word_index:
        num_words = len(self.word_index) + 1
      else:
        raise ValueError('Specify a dimension (`num_words` argument), '
                         'or fit on some text data first.')
    else:
      num_words = self.num_words

    if mode == 'tfidf' and not self.document_count:
      raise ValueError('Fit the Tokenizer on some data '
                       'before using tfidf mode.')

    x = np.zeros((len(sequences), num_words))
    for i, seq in enumerate(sequences):
      if not seq:
        continue
      counts = collections.defaultdict(int)
      for j in seq:
        if j >= num_words:
          continue
        counts[j] += 1
      for j, c in list(counts.items()):
        if mode == 'count':
          x[i][j] = c
        elif mode == 'freq':
          x[i][j] = c / len(seq)
        elif mode == 'binary':
          x[i][j] = 1
        elif mode == 'tfidf':
          # Use weighting scheme 2 in
          # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
          tf = 1 + np.log(c)
          idf = np.log(1 + self.document_count /
                       (1 + self.index_docs.get(j, 0)))
          x[i][j] = tf * idf
        else:
          raise ValueError('Unknown vectorization mode:', mode)
    return x

  def get_config(self):
    """Returns the tokenizer configuration as Python dictionary.

    The word count dictionaries used by the tokenizer get serialized
    into plain JSON, so that the configuration can be read by other
    projects.

    Returns:
        A Python dictionary with the tokenizer configuration.
    """
    json_word_counts = json.dumps(self.word_counts)
    json_word_docs = json.dumps(self.word_docs)
    json_index_docs = json.dumps(self.index_docs)
    json_word_index = json.dumps(self.word_index)
    json_index_word = json.dumps(self.index_word)

    return {
        'num_words': self.num_words,
        'filters': self.filters,
        'lower': self.lower,
        'split': self.split,
        'char_level': self.char_level,
        'oov_token': self.oov_token,
        'document_count': self.document_count,
        'word_counts': json_word_counts,
        'word_docs': json_word_docs,
        'index_docs': json_index_docs,
        'index_word': json_index_word,
        'word_index': json_word_index
    }

  def to_json(self, **kwargs):
    """Returns a JSON string containing the tokenizer configuration.

    To load a tokenizer from a JSON string, use
    `keras.preprocessing.text.tokenizer_from_json(json_string)`.

    Args:
        **kwargs: Additional keyword arguments
            to be passed to `json.dumps()`.

    Returns:
        A JSON string containing the tokenizer configuration.
    """
    config = self.get_config()
    tokenizer_config = {'class_name': self.__class__.__name__, 'config': config}
    return json.dumps(tokenizer_config, **kwargs)


@keras_export('keras.preprocessing.text.tokenizer_from_json')
def tokenizer_from_json(json_string):
  """Parses a JSON tokenizer configuration and returns a tokenizer instance.

  Deprecated: `tf.keras.preprocessing.text.Tokenizer` does not operate on
  tensors and is not recommended for new code. Prefer
  `tf.keras.layers.TextVectorization` which provides equivalent functionality
  through a layer which accepts `tf.Tensor` input. See the
  [text loading tutorial](https://www.tensorflow.org/tutorials/load_data/text)
  for an overview of the layer and text handling in tensorflow.

  Args:
      json_string: JSON string encoding a tokenizer configuration.

  Returns:
      A Keras Tokenizer instance
  """
  tokenizer_config = json.loads(json_string)
  config = tokenizer_config.get('config')

  word_counts = json.loads(config.pop('word_counts'))
  word_docs = json.loads(config.pop('word_docs'))
  index_docs = json.loads(config.pop('index_docs'))
  # Integer indexing gets converted to strings with json.dumps()
  index_docs = {int(k): v for k, v in index_docs.items()}
  index_word = json.loads(config.pop('index_word'))
  index_word = {int(k): v for k, v in index_word.items()}
  word_index = json.loads(config.pop('word_index'))

  tokenizer = Tokenizer(**config)
  tokenizer.word_counts = word_counts
  tokenizer.word_docs = word_docs
  tokenizer.index_docs = index_docs
  tokenizer.word_index = word_index
  tokenizer.index_word = index_word
  return tokenizer

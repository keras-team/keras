
## text_to_word_sequence

```python
keras.preprocessing.text.text_to_word_sequence(text,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
```

Split a sentence into a list of words.

- __Return__: List of words (str).

- __Arguments__:
    - __text__: str.
    - __filters__: list (or concatenation) of characters to filter out, such as
         punctuation. Default: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' , includes
         basic punctuation, tabs, and newlines.
    - __lower__: boolean. Whether to set the text to lowercase.
    - __split__: str. Separator for word splitting.

## one_hot

```python
keras.preprocessing.text.one_hot(text,
                                 n,
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                 lower=True,
                                 split=" ")
```

One-hot encodes a text into a list of word indexes in a vocabulary of size n.

This is a wrapper to the `hashing_trick` function using `hash` as the hashing function.

- __Return__: List of integers in [1, n]. Each integer encodes a word (unicity non-guaranteed).

- __Arguments__:
    - __text__: str.
    - __n__: int. Size of vocabulary.
    - __filters__: list (or concatenation) of characters to filter out, such as
         punctuation. Default: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' , includes
         basic punctuation, tabs, and newlines.
    - __lower__: boolean. Whether to set the text to lowercase.
    - __split__: str. Separator for word splitting.
    
## hashing_trick

```python
keras.preprocessing.text.hashing_trick(text, 
                                       n,
                                       hash_function=None,
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                       lower=True,
                                       split=' ')
```

Converts a text to a sequence of indices in a fixed-size hashing space

- __Return__:
        A list of integer word indices (unicity non-guaranteed).
- __Arguments__:
    - __text__: str.
    - __n__: Dimension of the hashing space.
    - __hash_function__: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
    - __filters__: list (or concatenation) of characters to filter out, such as
         punctuation. Default: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' , includes
         basic punctuation, tabs, and newlines.
    - __lower__: boolean. Whether to set the text to lowercase.
    - __split__: str. Separator for word splitting.

## Tokenizer

```python
keras.preprocessing.text.Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False,
                                   oov_token=None)
```

Class for vectorizing texts, or/and turning texts into sequences (=list of word indexes, where the word of rank i in the dataset (starting at 1) has index i).

- __Arguments__: Same as `text_to_word_sequence` above.
    - __num_words__: None or int. Maximum number of words to work with (if set, tokenization will be restricted to the top num_words most common words in the dataset).
    - __char_level__: if True, every character will be treated as a token.
    - __oov_token__: None or str. If given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls.

- __Methods__:

    - __fit_on_texts(texts)__: 
        - __Arguments__:
            - __texts__: list of texts to train on.

    - __texts_to_sequences(texts)__
        - __Arguments__: 
            - __texts__: list of texts to turn to sequences.
        - __Return__: list of sequences (one per text input).

    - __texts_to_sequences_generator(texts)__: generator version of the above. 
        - __Return__: yield one sequence per input text.

    - __texts_to_matrix(texts)__:
        - __Return__: numpy array of shape `(len(texts), num_words)`.
        - __Arguments__:
            - __texts__: list of texts to vectorize.
            - __mode__: one of "binary", "count", "tfidf", "freq" (default: "binary").

    - __fit_on_sequences(sequences)__: 
        - __Arguments__:
            - __sequences__: list of sequences to train on. 

    - __sequences_to_matrix(sequences)__:
        - __Return__: numpy array of shape `(len(sequences), num_words)`.
        - __Arguments__:
            - __sequences__: list of sequences to vectorize.
            - __mode__: one of "binary", "count", "tfidf", "freq" (default: "binary").

- __Attributes__:
    - __word_counts__: dictionary mapping words (str) to the number of times they appeared on during fit. Only set after fit_on_texts was called. 
    - __word_docs__: dictionary mapping words (str) to the number of documents/texts they appeared on during fit. Only set after fit_on_texts was called.
    - __word_index__: dictionary mapping words (str) to their rank/index (int). Only set after fit_on_texts was called.
    - __document_count__: int. Number of documents (texts/sequences) the tokenizer was trained on. Only set after fit_on_texts or fit_on_sequences was called.



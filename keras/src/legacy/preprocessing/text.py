"""Deprecated text preprocessing APIs from Keras 1."""

import collections
import hashlib
import json
import warnings

import numpy as np

from keras.src.api_export import keras_export


@keras_export("keras._legacy.preprocessing.text.text_to_word_sequence")
def text_to_word_sequence(
    input_text,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
):
    """DEPRECATED."""
    if lower:
        input_text = input_text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    input_text = input_text.translate(translate_map)

    seq = input_text.split(split)
    return [i for i in seq if i]


@keras_export("keras._legacy.preprocessing.text.one_hot")
def one_hot(
    input_text,
    n,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
    analyzer=None,
):
    """DEPRECATED."""
    return hashing_trick(
        input_text,
        n,
        hash_function=hash,
        filters=filters,
        lower=lower,
        split=split,
        analyzer=analyzer,
    )


@keras_export("keras._legacy.preprocessing.text.hashing_trick")
def hashing_trick(
    text,
    n,
    hash_function=None,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
    analyzer=None,
):
    """DEPRECATED."""
    if hash_function is None:
        hash_function = hash
    elif hash_function == "md5":

        def hash_function(w):
            return int(hashlib.md5(w.encode()).hexdigest(), 16)

    if analyzer is None:
        seq = text_to_word_sequence(
            text, filters=filters, lower=lower, split=split
        )
    else:
        seq = analyzer(text)

    return [(hash_function(w) % (n - 1) + 1) for w in seq]


@keras_export("keras._legacy.preprocessing.text.Tokenizer")
class Tokenizer:
    """DEPRECATED."""

    def __init__(
        self,
        num_words=None,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ",
        char_level=False,
        oov_token=None,
        analyzer=None,
        **kwargs,
    ):
        # Legacy support
        if "nb_words" in kwargs:
            warnings.warn(
                "The `nb_words` argument in `Tokenizer` "
                "has been renamed `num_words`."
            )
            num_words = kwargs.pop("nb_words")
        document_count = kwargs.pop("document_count", 0)
        if kwargs:
            raise TypeError(f"Unrecognized keyword arguments: {str(kwargs)}")

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
                        text,
                        filters=self.filters,
                        lower=self.lower,
                        split=self.split,
                    )
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
        self.word_index = dict(
            zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))
        )

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def fit_on_sequences(self, sequences):
        self.document_count += len(sequences)
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                self.index_docs[i] += 1

    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
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
                        text,
                        filters=self.filters,
                        lower=self.lower,
                        split=self.split,
                    )
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
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences):
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
            vect = " ".join(vect)
            yield vect

    def texts_to_matrix(self, texts, mode="binary"):
        sequences = self.texts_to_sequences(texts)
        return self.sequences_to_matrix(sequences, mode=mode)

    def sequences_to_matrix(self, sequences, mode="binary"):
        if not self.num_words:
            if self.word_index:
                num_words = len(self.word_index) + 1
            else:
                raise ValueError(
                    "Specify a dimension (`num_words` argument), "
                    "or fit on some text data first."
                )
        else:
            num_words = self.num_words

        if mode == "tfidf" and not self.document_count:
            raise ValueError(
                "Fit the Tokenizer on some data before using tfidf mode."
            )

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
                if mode == "count":
                    x[i][j] = c
                elif mode == "freq":
                    x[i][j] = c / len(seq)
                elif mode == "binary":
                    x[i][j] = 1
                elif mode == "tfidf":
                    # Use weighting scheme 2 in
                    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                    tf = 1 + np.log(c)
                    idf = np.log(
                        1
                        + self.document_count / (1 + self.index_docs.get(j, 0))
                    )
                    x[i][j] = tf * idf
                else:
                    raise ValueError("Unknown vectorization mode:", mode)
        return x

    def get_config(self):
        json_word_counts = json.dumps(self.word_counts)
        json_word_docs = json.dumps(self.word_docs)
        json_index_docs = json.dumps(self.index_docs)
        json_word_index = json.dumps(self.word_index)
        json_index_word = json.dumps(self.index_word)

        return {
            "num_words": self.num_words,
            "filters": self.filters,
            "lower": self.lower,
            "split": self.split,
            "char_level": self.char_level,
            "oov_token": self.oov_token,
            "document_count": self.document_count,
            "word_counts": json_word_counts,
            "word_docs": json_word_docs,
            "index_docs": json_index_docs,
            "index_word": json_index_word,
            "word_index": json_word_index,
        }

    def to_json(self, **kwargs):
        config = self.get_config()
        tokenizer_config = {
            "class_name": self.__class__.__name__,
            "config": config,
        }
        return json.dumps(tokenizer_config, **kwargs)


@keras_export("keras._legacy.preprocessing.text.tokenizer_from_json")
def tokenizer_from_json(json_string):
    """DEPRECATED."""
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get("config")

    word_counts = json.loads(config.pop("word_counts"))
    word_docs = json.loads(config.pop("word_docs"))
    index_docs = json.loads(config.pop("index_docs"))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop("index_word"))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop("word_index"))

    tokenizer = Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word
    return tokenizer

"""
Title: Pretraining BERT with Hugging Face Transformers
Author: Sreyan Ghosh
Date created: 2022/07/01
Last modified: 2022/08/27
Description: Pretraining BERT using Hugging Face Transformers on NSP and MLM.
Accelerator: GPU
"""

"""
## Introduction
"""

"""
### BERT (Bidirectional Encoder Representations from Transformers)

In the field of computer vision, researchers have repeatedly shown the value of
transfer learning — pretraining a neural network model on a known task/dataset, for
instance ImageNet classification, and then performing fine-tuning — using the trained neural
network as the basis of a new specific-purpose model. In recent years, researchers
have shown that a similar technique can be useful in many natural language tasks.

BERT makes use of Transformer, an attention mechanism that learns contextual relations
between words (or subwords) in a text. In its vanilla form, Transformer includes two
separate mechanisms — an encoder that reads the text input and a decoder that produces
a prediction for the task. Since BERT’s goal is to generate a language model, only the
encoder mechanism is necessary. The detailed workings of Transformer are described in
a paper by Google.

As opposed to directional models, which read the text input sequentially
(left-to-right or right-to-left), the Transformer encoder reads the entire
sequence of words at once. Therefore it is considered bidirectional, though
it would be more accurate to say that it’s non-directional. This characteristic
allows the model to learn the context of a word based on all of its surroundings
(left and right of the word).

When training language models, a challenge is defining a prediction goal.
Many models predict the next word in a sequence (e.g. `"The child came home from _"`),
a directional approach which inherently limits context learning. To overcome this
challenge, BERT uses two training strategies:

### Masked Language Modeling (MLM)

Before feeding word sequences into BERT, 15% of the words in each sequence are replaced
with a `[MASK]` token. The model then attempts to predict the original value of the masked
words, based on the context provided by the other, non-masked, words in the sequence.

### Next Sentence Prediction (NSP)

In the BERT training process, the model receives pairs of sentences as input and learns to
predict if the second sentence in the pair is the subsequent sentence in the original
document. During training, 50% of the inputs are a pair in which the second sentence is the
subsequent sentence in the original document, while in the other 50% a random sentence
from the corpus is chosen as the second sentence. The assumption is that the random sentence
will represent a disconnect from the first sentence.

Though Google provides a pretrained BERT checkpoint for English, you may often need
to either pretrain the model from scratch for a different language, or do a
continued-pretraining to fit the model to a new domain. In this notebook, we pretrain
BERT from scratch optimizing both MLM and NSP objectves using 🤗 Transformers on the `WikiText`
English dataset loaded from 🤗 Datasets.
"""

"""
## Setup
"""

"""
### Installing the requirements
"""

"""shell
pip install git+https://github.com/huggingface/transformers.git
pip install datasets
pip install huggingface-hub
pip install nltk
"""

"""
### Importing the necessary libraries
"""

import nltk
import random
import logging

import keras

nltk.download("punkt")
# Set random seed
keras.utils.set_random_seed(42)

"""
### Define certain variables
"""

TOKENIZER_BATCH_SIZE = 256  # Batch-size to train the tokenizer on
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

BLOCK_SIZE = 128  # Maximum number of tokens in an input sample
NSP_PROB = 0.50  # Probability that the next sentence is the actual next sentence in NSP
SHORT_SEQ_PROB = 0.1  # Probability of generating shorter sequences to minimize the mismatch between pretraining and fine-tuning.
MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

MLM_PROB = 0.2  # Probability with which tokens are masked in MLM

TRAIN_BATCH_SIZE = 2  # Batch-size for pretraining the model on
MAX_EPOCHS = 1  # Maximum number of epochs to train the model for
LEARNING_RATE = 1e-4  # Learning rate for training the model

MODEL_CHECKPOINT = "bert-base-cased"  # Name of pretrained model from 🤗 Model Hub

"""
## Load the WikiText dataset
"""

"""
We now download the `WikiText` language modeling dataset. It is a collection of over
100 million tokens extracted from the set of verified "Good" and "Featured" articles on
Wikipedia.

We load the dataset from [🤗 Datasets](https://github.com/huggingface/datasets).
For the purpose of demonstration in this notebook, we work with only the `train`
split of the dataset. This can be easily done with the `load_dataset` function.
"""

from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

"""
The dataset just has one column which is the raw text, and this is all we need for
pretraining BERT!
"""

print(dataset)

"""
## Training a new Tokenizer
"""

"""
First we train our own tokenizer from scratch on our corpus, so that can we
can use it to train our language model from scratch.

But why would you need to train a tokenizer? That's because Transformer models very
often use subword tokenization algorithms, and they need to be trained to identify the
parts of words that are often present in the corpus you are using.

The 🤗 Transformers `Tokenizer` (as the name indicates) will tokenize the inputs
(including converting the tokens to their corresponding IDs in the pretrained vocabulary)
and put it in a format the model expects, as well as generate the other inputs that model
requires.

First we make a list of all the raw documents from the `WikiText` corpus:
"""

all_texts = [
    doc for doc in dataset["train"]["text"] if len(doc) > 0 and not doc.startswith(" =")
]

"""
Next we make a `batch_iterator` function that will aid us to train our tokenizer.
"""


def batch_iterator():
    for i in range(0, len(all_texts), TOKENIZER_BATCH_SIZE):
        yield all_texts[i : i + TOKENIZER_BATCH_SIZE]


"""
In this notebook, we train a tokenizer with the exact same algorithms and
parameters as an existing one. For instance, we train a new version of the
`BERT-CASED` tokenzier on `Wikitext-2` using the same tokenization algorithm.

First we need to load the tokenizer we want to use as a model:
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

"""
Now we train our tokenizer using the entire `train` split of the `Wikitext-2`
dataset.
"""

tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(), vocab_size=TOKENIZER_VOCABULARY
)

"""
So now we our done training our new tokenizer! Next we move on to the data
pre-processing steps.
"""

"""
## Data Pre-processing
"""

"""
For the sake of demonstrating the workflow, in this notebook we only take
small subsets of the entire WikiText `train` and `test` splits.
"""

dataset["train"] = dataset["train"].select([i for i in range(1000)])
dataset["validation"] = dataset["validation"].select([i for i in range(1000)])

"""
Before we can feed those texts to our model, we need to pre-process them and get them
ready for the task. As mentioned earlier, the BERT pretraining task includes two tasks
in total, the `NSP` task and the `MLM` task. 🤗 Transformers have an easy to implement
`collator` called the `DataCollatorForLanguageModeling`. However, we need to get the
data ready for `NSP` manually.

Next we write a simple function called the `prepare_train_features` that helps us in
the pre-processing and is compatible with 🤗 Datasets. To summarize, our pre-processing
function should:

- Get the dataset ready for the NSP task by creating pairs of sentences (A,B), where B
either actually follows A, or B is randomly sampled from somewhere else in the corpus.
It should also generate a corresponding label for each pair, which is 1 if B actually
follows A and 0 if not.
- Tokenize the text dataset into it's corresponding token ids that will be used for
embedding look-up in BERT
- Create additional inputs for the model like `token_type_ids`, `attention_mask`, etc.
"""

# We define the maximum number of tokens after tokenization that each training sample
# will have
max_num_tokens = BLOCK_SIZE - tokenizer.num_special_tokens_to_add(pair=True)


def prepare_train_features(examples):
    """Function to prepare features for NSP task

    Arguments:
      examples: A dictionary with 1 key ("text")
        text: List of raw documents (str)
    Returns:
      examples:  A dictionary with 4 keys
        input_ids: List of tokenized, concatnated, and batched
          sentences from the individual raw documents (int)
        token_type_ids: List of integers (0 or 1) corresponding
          to: 0 for senetence no. 1 and padding, 1 for sentence
          no. 2
        attention_mask: List of integers (0 or 1) corresponding
          to: 1 for non-padded tokens, 0 for padded
        next_sentence_label: List of integers (0 or 1) corresponding
          to: 1 if the second sentence actually follows the first,
          0 if the senetence is sampled from somewhere else in the corpus
    """

    # Remove un-wanted samples from the training set
    examples["document"] = [
        d.strip() for d in examples["text"] if len(d) > 0 and not d.startswith(" =")
    ]
    # Split the documents from the dataset into it's individual sentences
    examples["sentences"] = [
        nltk.tokenize.sent_tokenize(document) for document in examples["document"]
    ]
    # Convert the tokens into ids using the trained tokenizer
    examples["tokenized_sentences"] = [
        [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) for sent in doc]
        for doc in examples["sentences"]
    ]

    # Define the outputs
    examples["input_ids"] = []
    examples["token_type_ids"] = []
    examples["attention_mask"] = []
    examples["next_sentence_label"] = []

    for doc_index, document in enumerate(examples["tokenized_sentences"]):
        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.
        target_seq_length = max_num_tokens

        if random.random() < SHORT_SEQ_PROB:
            target_seq_length = random.randint(2, max_num_tokens)

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    if len(current_chunk) == 1 or random.random() < NSP_PROB:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = random.randint(
                                0, len(examples["tokenized_sentences"]) - 1
                            )
                            if random_document_index != doc_index:
                                break

                        random_document = examples["tokenized_sentences"][
                            random_document_index
                        ]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    input_ids = tokenizer.build_inputs_with_special_tokens(
                        tokens_a, tokens_b
                    )
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(
                        tokens_a, tokens_b
                    )

                    padded = tokenizer.pad(
                        {"input_ids": input_ids, "token_type_ids": token_type_ids},
                        padding="max_length",
                        max_length=MAX_LENGTH,
                    )

                    examples["input_ids"].append(padded["input_ids"])
                    examples["token_type_ids"].append(padded["token_type_ids"])
                    examples["attention_mask"].append(padded["attention_mask"])
                    examples["next_sentence_label"].append(1 if is_random_next else 0)
                    current_chunk = []
                    current_length = 0
            i += 1

    # We delete all the un-necessary columns from our dataset
    del examples["document"]
    del examples["sentences"]
    del examples["text"]
    del examples["tokenized_sentences"]

    return examples


tokenized_dataset = dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=["text"],
    num_proc=1,
)

"""
For MLM we are going to use the same preprocessing as before for our dataset with
one additional step: we randomly mask some tokens (by replacing them by [MASK])
and the labels will be adjusted to only include the masked tokens
(we don't have to predict the non-masked tokens). If you use a tokenizer you trained
yourself, make sure the [MASK] token is among the special tokens you passed during training!

To get the data ready for MLM, we simply use the `collator` called the
`DataCollatorForLanguageModeling` provided by the 🤗 Transformers library on our dataset
that is already ready for the NSP task. The `collator` expects certain parameters.
We use the default ones from the original BERT paper in this notebook. The
`return_tensors='tf'` ensures that we get `tf.Tensor` objects back.
"""

from transformers import DataCollatorForLanguageModeling

collater = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB, return_tensors="tf"
)

"""
Next we define our training set with which we train our model. Again, 🤗 Datasets
provides us with the `to_tf_dataset` method which will help us integrate our dataset with
the `collator` defined above. The method expects certain parameters:

- **columns**: the columns which will serve as our independant variables
- **label_cols**: the columns which will serve as our labels or dependant variables
- **batch_size**: our batch size for training
- **shuffle**: whether we want to shuffle our training dataset
- **collate_fn**: our collator function
"""

train = tokenized_dataset["train"].to_tf_dataset(
    columns=["input_ids", "token_type_ids", "attention_mask"],
    label_cols=["labels", "next_sentence_label"],
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collater,
)

validation = tokenized_dataset["validation"].to_tf_dataset(
    columns=["input_ids", "token_type_ids", "attention_mask"],
    label_cols=["labels", "next_sentence_label"],
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collater,
)

"""
## Defining the model
"""

"""
To define our model, first we need to define a config which will help us define certain
parameters of our model architecture. This includes parameters like number of transformer
layers, number of attention heads, hidden dimension, etc. For this notebook, we try
to define the exact config defined in the original BERT paper.

We can easily achieve this using the `BertConfig` class from the 🤗 Transformers library.
The `from_pretrained()` method expects the name of a model. Here we define the simplest
model with which we also trained our model, i.e., `bert-base-cased`.
"""

from transformers import BertConfig

config = BertConfig.from_pretrained(MODEL_CHECKPOINT)

"""
For defining our model we use the `TFBertForPreTraining` class from the 🤗 Transformers
library. This class internally handles everything starting from defining our model, to
unpacking our inputs and calculating the loss. So we need not do anything ourselves except
defining the model with the correct `config` we want!
"""

from transformers import TFBertForPreTraining

model = TFBertForPreTraining(config)

"""
Now we define our optimizer and compile the model. The loss calculation is handled
internally and so we need not worry about that!
"""

from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE))

"""
Finally all steps are done and now we can start training our model!
"""

model.fit(train, validation_data=validation, epochs=MAX_EPOCHS)

"""
Our model has now been trained! We suggest to please train the model on the complete
dataset for atleast 50 epochs for decent performance. The pretrained model now acts as
a language model and is meant to be fine-tuned on a downstream task. Thus it can now be
fine-tuned on any downstream task like Question Answering, Text Classification
etc.!
"""

"""
Now you can push this model to 🤗 Model Hub and also share it with with all your friends,
family, favorite pets: they can all load it with the identifier
`"your-username/the-name-you-picked"` so for instance:

```python
model.push_to_hub("pretrained-bert", organization="keras-io")
tokenizer.push_to_hub("pretrained-bert", organization="keras-io")
```
And after you push your model this is how you can load it in the future!

```python
from transformers import TFBertForPreTraining

model = TFBertForPreTraining.from_pretrained("your-username/my-awesome-model")
```
or, since it's a pretrained model and you would generally use it for fine-tuning
on a downstream task, you can also load it for some other task like:

```python
from transformers import TFBertForSequenceClassification

model = TFBertForSequenceClassification.from_pretrained("your-username/my-awesome-model")
```
In this case, the pretraining head will be dropped and the model will just be initialized
with the transformer layers. A new task-specific head will be added with random weights.
"""
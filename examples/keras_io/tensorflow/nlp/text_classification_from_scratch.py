"""
Title: Text classification from scratch
Authors: Mark Omernick, Francois Chollet
Date created: 2019/11/06
Last modified: 2020/05/17
Description: Text sentiment classification starting from raw text files.
Accelerator: GPU
"""
"""
## Introduction

This example shows how to do text classification starting from raw text (as
a set of text files on disk). We demonstrate the workflow on the IMDB sentiment
classification dataset (unprocessed version). We use the `TextVectorization` layer for
 word splitting & indexing.
"""

"""
## Setup
"""

import tensorflow as tf
import keras
from keras.layers import TextVectorization
from keras import layers
import string
import re
import os
from pathlib import Path

"""
## Load the data: IMDB movie review sentiment classification

Let's download the data and inspect its structure.
"""

fpath = keras.utils.get_file(
    origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
)
dirpath = Path(fpath).parent.absolute()
os.system(f"tar -xf {fpath} -C {dirpath}")


"""
The `aclImdb` folder contains a `train` and `test` subfolder:
"""

os.system(f"ls {dirpath}/aclImdb")
os.system(f"ls {dirpath}/aclImdb/train")
os.system(f"ls {dirpath}/aclImdb/test")

"""
The `aclImdb/train/pos` and `aclImdb/train/neg` folders contain text files, each of
 which represents one review (either positive or negative):
"""

os.system(f"cat {dirpath}/aclImdb/train/pos/6248_7.txt")

"""
We are only interested in the `pos` and `neg` subfolders, so let's delete the rest:
"""

os.system(f"rm -r {dirpath}/aclImdb/train/unsup")


"""
You can use the utility `keras.utils.text_dataset_from_directory` to
generate a labeled `tf.data.Dataset` object from a set of text files on disk filed
 into class-specific folders.

Let's use it to generate the training, validation, and test datasets. The validation
and training datasets are generated from two subsets of the `train` directory, with 20%
of samples going to the validation dataset and 80% going to the training dataset.

Having a validation dataset in addition to the test dataset is useful for tuning
hyperparameters, such as the model architecture, for which the test dataset should not
be used.

Before putting the model out into the real world however, it should be retrained using all
available training data (without creating a validation dataset), so its performance is maximized.

When using the `validation_split` & `subset` arguments, make sure to either specify a
random seed, or to pass `shuffle=False`, so that the validation & training splits you
get have no overlap.
"""

batch_size = 32
raw_train_ds, raw_val_ds = keras.utils.text_dataset_from_directory(
    f"{dirpath}/aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="both",
    seed=1337,
)
raw_test_ds = keras.utils.text_dataset_from_directory(
    f"{dirpath}/aclImdb/test", batch_size=batch_size
)

print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")

"""
Let's preview a few samples:
"""

# It's important to take a look at your raw data to ensure your normalization
# and tokenization will work as expected. We can do that by taking a few
# examples from the training set and looking at them.
# This is one of the places where eager execution shines:
# we can just evaluate these tensors using .numpy()
# instead of needing to evaluate them in a Session/Graph context.
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])

"""
## Prepare the data

In particular, we remove `<br />` tags.
"""


# Having looked at our data above, we see that the raw text contains HTML break
# tags of the form '<br />'. These tags will not be removed by the default
# standardizer (which doesn't strip HTML). Because of this, we will need to
# create a custom standardization function.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


# Model constants.
max_features = 20000
embedding_dim = 128
sequence_length = 500

# Now that we have our custom standardization, we can instantiate our text
# vectorization layer. We are using this layer to normalize, split, and map
# strings to integers, so we set our 'output_mode' to 'int'.
# Note that we're using the default split function,
# and the custom standardization defined above.
# We also set an explicit maximum sequence length, since the CNNs later in our
# model won't support ragged sequences.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Now that the vocab layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary. You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(text_ds)

"""
## Two options to vectorize the data

There are 2 ways we can use our text vectorization layer:

**Option 1: Make it part of the model**, so as to obtain a model that processes raw
strings, like this:
"""

"""

```python
text_input = keras.Input(shape=(1,), dtype=tf.string, name='text')
x = vectorize_layer(text_input)
x = layers.Embedding(max_features + 1, embedding_dim)(x)
...
```

**Option 2: Apply it to the text dataset** to obtain a dataset of word indices, then
 feed it into a model that expects integer sequences as inputs.

An important difference between the two is that option 2 enables you to do
**asynchronous CPU processing and buffering** of your data when training on GPU.
So if you're training the model on GPU, you probably want to go with this option to get
the best performance. This is what we will do below.

If we were to export our model to production, we'd ship a model that accepts raw
strings as input, like in the code snippet for option 1 above. This can be done after
training. We do this in the last section.
"""


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

"""
## Build a model

We choose a simple 1D convnet starting with an `Embedding` layer.
"""

# A integer input for vocab indices.
inputs = keras.Input(shape=(sequence_length,), dtype="int64")
# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = keras.layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
model = keras.Model(inputs, predictions)
model.summary()
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)


"""
## Train the model
"""

epochs = 3

# Fit the model using the train and test datasets.
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

"""
## Evaluate the model on the test set
"""

model.evaluate(test_ds)

"""
## Make an end-to-end model

If you want to obtain a model capable of processing raw strings, you can simply
create a new model (using the weights we just trained):
"""

# A string input
inputs = keras.Input(shape=(1,), dtype="string")
# Turn strings into vocab indices
indices = vectorize_layer(inputs)
# Turn vocab indices into predictions
outputs = model(indices)

# Our end to end model
end_to_end_model = keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Test it with `raw_test_ds`, which yields raw strings
end_to_end_model.evaluate(raw_test_ds)

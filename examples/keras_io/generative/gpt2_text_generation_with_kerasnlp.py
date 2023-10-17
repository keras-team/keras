"""
Title: GPT2 Text Generation with KerasNLP
Author: Chen Qian
Date created: 04/17/2023
Last modified: 04/17/2023
Description: Use KerasNLP GPT2 model and `samplers` to do text generation.
Accelerator: GPU
"""

"""
In this tutorial, you will learn to use [KerasNLP](https://keras.io/keras_nlp/) to load a
pre-trained Large Language Model (LLM) - [GPT-2 model](https://openai.com/research/better-language-models)
(originally invented by OpenAI), finetune it to a specific text style, and
generate text based on users' input (also known as prompt). You will also learn
how GPT2 adapts quickly to non-English languages, such as Chinese.
"""

"""
##  Before we begin

Colab offers different kinds of runtimes. Make sure to go to **Runtime ->
Change runtime type** and choose the GPU Hardware Accelerator runtime
(which should have >12G host RAM and ~15G GPU RAM) since you will finetune the
GPT-2 model. Running this tutorial on CPU runtime will take hours.
"""

"""
## Install KerasNLP, Choose Backend and Import Dependencies

This examples uses [Keras 3.0](https://keras.io/keras_core/) to work in any of
`"tensorflow"`, `"jax"` or `"torch"`. Support for Keras Core is baked into
KerasNLP, simply change the `"KERAS_BACKEND"` environment variable to select
the backend of your choice. We select the JAX backend below.
"""

"""shell
pip install git+https://github.com/keras-team/keras-nlp.git -q
"""

import keras_nlp
import tensorflow as tf
import json
import keras
import os
import tensorflow_datasets as tfds
import time


"""
## Introduction to Generative Large Language Models (LLMs)

Large language models (LLMs) are a type of machine learning models that are
trained on a large corpus of text data to generate outputs for various natural
language processing (NLP) tasks, such as text generation, question answering,
and machine translation.

Generative LLMs are typically based on deep learning neural networks, such as
the [Transformer architecture](https://arxiv.org/abs/1706.03762) invented by
Google researchers in 2017, and are trained on massive amounts of text data,
often involving billions of words. These models, such as Google [LaMDA](https://blog.google/technology/ai/lamda/)
and [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html),
are trained with a large dataset from various data sources which allows them to
generate output for many tasks. The core of Generative LLMs is predicting the
next word in a sentence, often referred as **Causal LM Pretraining**. In this
way LLMs can generate coherent text based on user prompts. For a more
pedagogical discussion on language models, you can refer to the
[Stanford CS324 LLM class](https://stanford-cs324.github.io/winter2022/lectures/introduction/).
"""

"""
## Introduction to KerasNLP

Large Language Models are complex to build and expensive to train from scratch.
Luckily there are pretrained LLMs available for use right away. [KerasNLP](https://keras.io/keras_nlp/)
provides a large number of pre-trained checkpoints that allow you to experiment
with SOTA models without needing to train them yourself.

KerasNLP is a natural language processing library that supports users through
their entire development cycle. KerasNLP offers both pretrained models and
modularized building blocks, so developers could easily reuse pretrained models
or stack their own LLM.

In a nutshell, for generative LLM, KerasNLP offers:

- Pretrained models with `generate()` method, e.g.,
    `keras_nlp.models.GPT2CausalLM` and `keras_nlp.models.OPTCausalLM`.
- Sampler class that implements generation algorithms such as Top-K, Beam and
    contrastive search. These samplers can be used to generate text with
    custom models.
"""

"""
## Load a pre-trained GPT-2 model and generate some text

KerasNLP provides a number of pre-trained models, such as [Google
Bert](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
and [GPT-2](https://openai.com/research/better-language-models). You can see
the list of models available in the [KerasNLP repository](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/models).

It's very easy to load the GPT-2 model as you can see below:
"""

# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)

"""
Once the model is loaded, you can use it to generate some text right away. Run
the cells below to give it a try. It's as simple as calling a single function
*generate()*:
"""

start = time.time()

output = gpt2_lm.generate("My trip to Yosemite was", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

"""
Try another one:
"""

start = time.time()

output = gpt2_lm.generate("That Italian restaurant is", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

"""
Notice how much faster the second call is. This is because the computational
graph is [XLA compiled](https://www.tensorflow.org/xla) in the 1st run and
re-used in the 2nd behind the scenes.

The quality of the generated text looks OK, but we can improve it via
fine-tuning.
"""

"""
## More on the GPT-2 model from KerasNLP

Next up, we will actually fine-tune the model to update its parameters, but
before we do, let's take a look at the full set of tools we have to for working
with for GPT2.

The code of GPT2 can be found
[here](https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/models/gpt2/).
Conceptually the `GPT2CausalLM` can be hierarchically broken down into several
modules in KerasNLP, all of which have a *from_preset()* function that loads a
pretrained model:

- `keras_nlp.models.GPT2Tokenizer`: The tokenizer used by GPT2 model, which is a
    [byte-pair encoder](https://huggingface.co/course/chapter6/5?fw=pt).
- `keras_nlp.models.GPT2CausalLMPreprocessor`: the preprocessor used by GPT2
    causal LM training. It does the tokenization along with other preprocessing
    works such as creating the label and appending the end token.
- `keras_nlp.models.GPT2Backbone`: the GPT2 model, which is a stack of
    `keras_nlp.layers.TransformerDecoder`. This is usually just referred as
    `GPT2`.
- `keras_nlp.models.GPT2CausalLM`: wraps `GPT2Backbone`, it multiplies the
    output of `GPT2Backbone` by embedding matrix to generate logits over
    vocab tokens.
"""

"""
## Finetune on Reddit dataset

Now you have the knowledge of the GPT-2 model from KerasNLP, you can take one
step further to finetune the model so that it generates text in a specific
style, short or long, strict or casual. In this tutorial, we will use reddit
dataset for example.
"""

reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)

"""
Let's take a look inside sample data from the reddit TensorFlow Dataset. There
are two features:

- **__document__**: text of the post.
- **__title__**: the title.

"""

for document, title in reddit_ds:
    print(document.numpy())
    print(title.numpy())
    break

"""
In our case, we are performing next word prediction in a language model, so we
only need the 'document' feature.
"""

train_ds = (
    reddit_ds.map(lambda document, _: document)
    .batch(32)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

"""
Now you can finetune the model using the familiar *fit()* function. Note that
`preprocessor` will be automatically called inside `fit` method since
`GPT2CausalLM` is a `keras_nlp.models.Task` instance.

This step takes quite a bit of GPU memory and a long time if we were to train
it all the way to a fully trained state. Here we just use part of the dataset
for demo purposes.
"""

train_ds = train_ds.take(500)
num_epochs = 1

# Linearly decaying learning rate.
learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-5,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)

"""
After fine-tuning is finished, you can again generate text using the same
*generate()* function. This time, the text will be closer to Reddit writing
style, and the generated length will be close to our preset length in the
training set.
"""

start = time.time()

output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

"""
## Into the Sampling Method

In KerasNLP, we offer a few sampling methods, e.g., contrastive search,
Top-K and beam sampling. By default, our `GPT2CausalLM` uses Top-k search, but
you can choose your own sampling method.

Much like optimizer and activations, there are two ways to specify your custom
sampler:

- Use a string identifier, such as "greedy", you are using the default
configuration via this way.
- Pass a `keras_nlp.samplers.Sampler` instance, you can use custom configuration
via this way.
"""

# Use a string identifier.
gpt2_lm.compile(sampler="top_k")
output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

# Use a `Sampler` instance. `GreedySampler` tends to repeat itself,
greedy_sampler = keras_nlp.samplers.GreedySampler()
gpt2_lm.compile(sampler=greedy_sampler)

output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

"""
For more details on KerasNLP `Sampler` class, you can check the code
[here](https://github.com/keras-team/keras-nlp/tree/master/keras_nlp/samplers).
"""

"""
## Finetune on Chinese Poem Dataset

We can also finetune GPT2 on non-English datasets. For readers knowing Chinese,
this part illustrates how to fine-tune GPT2 on Chinese poem dataset to teach our
model to become a poet!

Because GPT2 uses byte-pair encoder, and the original pretraining dataset
contains some Chinese characters, we can use the original vocab to finetune on
Chinese dataset.
"""

"""shell
# Load chinese poetry dataset.
git clone https://github.com/chinese-poetry/chinese-poetry.git
"""

"""
Load text from the json file. We only use《全唐诗》for demo purposes.
"""

poem_collection = []
for file in os.listdir("chinese-poetry/全唐诗"):
    if ".json" not in file or "poet" not in file:
        continue
    full_filename = f"{'chinese-poetry/全唐诗'}/{file}"
    with open(full_filename, "r") as f:
        content = json.load(f)
        poem_collection.extend(content)

paragraphs = ["".join(data["paragraphs"]) for data in poem_collection]

"""
Let's take a look at sample data.
"""

print(paragraphs[0])

"""
Similar as Reddit example, we convert to TF dataset, and only use partial data
to train.
"""

train_ds = (
    tf.data.Dataset.from_tensor_slices(paragraphs)
    .batch(16)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# Running through the whole dataset takes long, only take `500` and run 1
# epochs for demo purposes.
train_ds = train_ds.take(500)
num_epochs = 1

learning_rate = keras.optimizers.schedules.PolynomialDecay(
    5e-4,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=loss,
    weighted_metrics=["accuracy"],
)

gpt2_lm.fit(train_ds, epochs=num_epochs)

"""
Let's check the result!
"""

output = gpt2_lm.generate("昨夜雨疏风骤", max_length=200)
print(output)

"""
Not bad 😀
"""

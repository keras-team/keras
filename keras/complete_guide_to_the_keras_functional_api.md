
## Simple fully connected network

THe `Sequential` model is probably a better choice to implement such a network, but it helps to start with something really simple.

```python
from keras.layers import Input, Dense
from keras.models import Model

# this returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# this creates a model that includes
# the Input layer and three Dense layers
model = Model(input=inputs, output=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)
```

As you can see, a layer instance is callable (on a tensor), and it returns a tensor. Input tensor(s) and output tensor(s) can then be used to define a model, which can be trained just like `Sequential` models.

## Multi-input and multi-output models

A good use case for the functional API: models with multiple inputs and outputs. The functional API makes it really easy to manipulate a large number of intertwinned datastreams.

Let's consider a model that learns a question answering model, that learns a relevance embedding space where questions and their answers will be embedded at closed positions (which will allow us to quickly query a database of answers to find those that are relevant to a new question).

The model has three input branches: an embedding for the question, and two embeddings for two different answers, a relevant answer and an unrelated answer. We'll train the model with a triplet loss, teaching the model to maximize the dot product between the question embedding and the embedding for the relevant answer, while minimizing the dot product between the question and the irrelevant answer.

[model graph]

Implementing this with the functional API is quick and simple:

```python

# an input question will be a vector of 100 integers,
# each being the index of a word in a vocabulary
question_input = Input(shape=(100,), dtype='int32')

good_answer_input = Input(shape=(100,), dtype='int32')
bad_answer_input = Input(shape=(100,), dtype='int32')

embedded_question = Embedding(output_dim=1024)(question_input)
encoded_question = LSTM(512)(embedded_question)

# these two layers will be shared across the two answer inputs.
# we'll go over shared layers in detail in the next section.
answer_embedding = Embedding(output_dim=1024)
answer_lstm = LSTM(512)

embedded_good_answer = answer_embedding(good_answer_input)
encoded_good_answer = answer_lstm(embedded_good_answer)

embedded_bad_answer = answer_embedding(bad_answer_input)
encoded_bad_answer = answer_lstm(embedded_bad_answer)

# let's take the dot product between the question embedding
# and the embedding of the answers
good_answer_score = merge([encoded_question, encoded_good_answer], mode='dot')
good_answer_score = merge([encoded_question, encoded_bad_answer], mode='dot')

# this is a lambda layer. It allows you to create
# simple stateless layers on the fly to take care of basic operations.
# Note how the layer below has multiple inputs.
output = Lambda(lambda x, y: x - y)([good_answer_score, good_answer_score])
```

Now let's say that we want out model to return not only the final output, but also each of the two previous scores (so you can apply auxilliary loss functions to them).
You can define the following model:

```python
model = Model(input=[question_input, good_answer_input, bad_answer_input],
              output=[output, good_answer_score, bad_answer_score])
model.compile(optimizer='rmsprop', loss=[custom_loss_1, custom_loss_2, custom_loss_3])
model.fit([q_data, good_ans_data, bad_ans_data], [custom_target_1, custom_target_2, custom_target_3])
```

You can also define a separate model that just embeds a single question, using the same layers as trained with the model above:
```python
question_embedder = Model(input=question_input, output=encoded_question)
embedded_qs = question_embedder.predict(q_data)
```
And one that can embed any answer:
```python
answer_embedder = Model(input=good_answer_input, output=encoded_good_answer)
embedded_ans = answer_embedder.predict(ans_data)
```

Great. Now if you have some training data, you got yourself a question/answer matching model.

## Shared layers

Another good use case for the functional API: models that use shared layers. Let's take a look at shared layers.

Let's consider a dataset of tweets. We want to build a model that can tell whether two tweets are from the same person or not (this can allow us to compare users by similarity, for instance).

One way to achieve this is to build a model that encodes two tweets into two vectors, concatenates the vectors and adds a logistic regression of top, outputting a probability that the tweets share the same author. The model would then be trained on positive tweet pairs and negative tweet pairs.

Because the problem is symetric, the mechanism that encodes the first tweet should be reused (weights and all) to encode the second tweet, as such:

(graph: shared lstm)

Here we use a LSTM layer to encode the tweets.

Let's build this with the functional API. We will take as input for a tweet a binary matrix of shape `(140, 256)`, i.e. a sequence of 140 vectors of size 256, where each dimension in the 256-dimensional vector encodes the presence/absence of a character (out of an alphabet of 256 frequent character).

```python
from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))
```

Sharing a layer across different inputs is easy: simply instantiate the layer once, then call it on as many inputs as you want:

```
# this layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64)

# when we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# we can then concatenate the two vectors:
merged_vector = merge([encoded_a, encoded_b], mode='concat', concat_axis=-1)

# and add a logistic regression on top
predictions = Dense(1, activation='sigmoid')(merged_vector) 

# we define a trainable model linking the
# tweet inputs to the predictions
model = Model(input=[tweet_a, tweet_b], output=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, nb_epoch=10)
```

Let's pause to take a look at how to read the shared layer's output or output shape.

## The concept of layer "node"

Whenever you are calling a layer on some input, you are creating a new tensor (the output of the layer), and you are creating adding a "node" to the layer, linking the input tensor to the output tensor. When you are calling a same layer multiple times, the layer owns multiple nodes, indexed as 0, 1, 2...

In previous versions of Keras, you could obtain the output tensor of a layer instance via `layer.get_output()`, or its output shape via `layer.output_shape`. You still can (except `get_output()` has been replaced by the property `output`). But what if a layer is connected to multiple inputs?

As long as a layer is only connected to one input, there is no confusion, and `.output` will return the one output of the layer:

```python
a = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```

Not so if the layer has multiple inputs:
```python
a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```
```
AssertionError: Layer lstm_4541336528 has multiple inbound nodes, hence the notion of "layer output" is ill-defined. Use `get_output_at(node_index)` instead.
```

Okay then. The following works:
```python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

Simple enough, right?

The same is true for `input_shape` and `output_shape`: as long as the layer has only one node, or as long as all nodes have the same input/output shape, then the notion of "layer output/input shape" is well defined, and that one shape will be returned by `layer.output_shape`/`layer.input_shape`. But if, for instance, you apply a same `Convolution2D` layer to an input of shape (3, 32, 32) then to an input of shape (3, 64, 64), the layer will have multiple input/output shapes, and you will have to fetch them via the index of the node they belong to:

```python
a = Input(shape=(3, 32, 32))
b = Input(shape=(3, 64, 64))

conv = Convolution2D(16, 3, 3, border_mode='same')
conved_a = conv(a)

# only one input so far, the following will work:
assert conv.input_shape == (None, 3, 32, 32)

conved_b = conv(b)
# now the `.input_shape` wouldn't work, but this does:
assert conv.get_input_shape_at(0) == (None, 3, 32, 32)
assert conv.get_input_shape_at(1) == (None, 3, 64, 64)
```





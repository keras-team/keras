"""
Title: Approximating non-Function Mappings with Mixture Density Networks
Author: [lukewood](https://twitter.com/luke_wood_ml)
Date created: 2023/07/15
Last modified: 2024/05/19
Description: Approximate non one to one mapping using mixture density networks.
Accelerator: None
"""

"""
## Approximating NonFunctions

Neural networks are universal function approximators. Key word: function!
While powerful function approximators, neural networks are not able to
approximate non-functions.
One important characteristic of functions is that they map one input to a
unique output.
Neural networks do not perform well when the training set has multiple values of
Y for a single X.
Instead of learning the proper distribution, a naive neural network will
interpret the problem as a function and learn the geometric mean of all `Y` in
the training set.

In this guide I'll show you how to approximate the class of non-functions
consisting of mappings from `x -> y` such that multiple `y` may exist for a
given `x`.  We'll use a class of neural networks called
"Mixture Density Networks".

I'm going to use the new
[Keras Core project](https://github.com/keras-team/keras) to
build my Mixture Density networks.
Great job to the Keras team on the project - it's awesome to be able to swap
frameworks in one line of code.

Some bad news: I use TensorFlow probability in this guide... so it actually work
only with JAX or TensorFlow backends.

Anyways, let's start by installing dependencies and sorting out imports:
"""
"""shell
pip install -q tensorflow-probability==0.24.0 keras==3.3.3 jax[cpu]==0.4.28 jaxlib==0.4.28
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import warnings
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import callbacks, layers, ops

# Replace with `from tensorflow_probability import distributions as tfd` for TensorFlow backend
from tensorflow_probability.substrates.jax import distributions as tfd


"""
Next, lets generate a noisy spiral that we're going to attempt to approximate.
I've defined a few functions below to do this:
"""


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def create_noisy_spiral(n, jitter_std=0.2, revolutions=2):
    angle = np.random.uniform(0, 2 * np.pi * revolutions, [n])
    r = angle

    x = r * np.cos(angle)
    y = r * np.sin(angle)

    result = np.stack([x, y], axis=1)
    result = result + np.random.normal(scale=jitter_std, size=[n, 2])
    result = 5 * normalize(result)
    return result


"""
Next, lets invoke this function many times to construct a sample dataset:
"""

xy = create_noisy_spiral(10000)

x, y = xy[:, 0:1], xy[:, 1:]

plt.scatter(x, y)
plt.show()

"""
As you can see, there's multiple possible values for Y with respect to a given
X.
Normal neural networks will simply learn the mean of these points with
respect to geometric space.
In the context of our spiral, however, the geometric mean of the each Y occurs
with a probability of zero.

We can quickly show this with a simple linear model:
"""

N_HIDDEN = 128

model = keras.Sequential(
    [
        layers.Dense(N_HIDDEN, activation="relu"),
        layers.Dense(N_HIDDEN, activation="relu"),
        layers.Dense(1),
    ]
)

"""
Let's use mean squared error as well as the adam optimizer.
These tend to be reasonable prototyping choices:
"""

model.compile(optimizer="adam", loss="mse")

"""
We can fit this model quite easy
"""

model.fit(
    x,
    y,
    epochs=300,
    batch_size=128,
    validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=10)],
)

"""
And let's check out the result:
"""

y_pred = model.predict(x)

"""
As expected, the model learns the geometric mean of all points in `y` for a
given `x`.
"""

plt.scatter(x, y)
plt.scatter(x, y_pred)
plt.show()

"""

## Mixture Density Networks

Mixture Density networks can alleviate this problem.
A mixture density is a class of complicated densities expressible in terms of simpler densities.
Effectively, a mixture density is the sum of various probability distributions.
By summing various distributions, mixture density distributions can
model arbitrarily complex distributions.
Mixture Density networks learn to parameterize a mixture density distribution
based on a given training set.

As a practitioner, all you need to know, is that Mixture Density Networks solve
the problem of multiple values of Y for a given X.
I'm hoping to add a tool to your kit- but I'm not going to formally explain the
derivation of Mixture Density networks in this guide.
The most important thing to know is that a Mixture Density network learns to
parameterize a mixture density distribution.
This is done by computing a special loss with respect to both the provided
`y_i` label as well as the predicted distribution for the corresponding `x_i`.
This loss function operates by computing the probability that `y_i` would be
drawn from the predicted mixture distribution.

Let's implement a Mixture density network.
Below, a ton of helper functions are defined based on an old Keras library
[`Keras Mixture Density Network Layer`](https://github.com/cpmpercussion/keras-mdn-layer).

I've adapted the code for use with Keras V3.

Lets start writing a Mixture Density Network!
First, we need a special activation function: ELU plus a tiny epsilon.
This helps prevent ELU from outputting 0 which causes NaNs in Mixture Density
Network loss evaluation.
"""


def elu_plus_one_plus_epsilon(x):
    return keras.activations.elu(x) + 1 + keras.backend.epsilon()


"""
Next, lets actually define a MixtureDensity layer that outputs all values needed
to sample from the learned mixture distribution:
"""


class MixtureDensityOutput(layers.Layer):
    def __init__(self, output_dimension, num_mixtures, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.mdn_mus = layers.Dense(
            self.num_mix * self.output_dim, name="mdn_mus"
        )  # mix*output vals, no activation
        self.mdn_sigmas = layers.Dense(
            self.num_mix * self.output_dim,
            activation=elu_plus_one_plus_epsilon,
            name="mdn_sigmas",
        )  # mix*output vals exp activation
        self.mdn_pi = layers.Dense(
            self.num_mix, name="mdn_pi"
        )  # mix vals, logits

    def call(self, x, mask=None):
        return layers.concatenate(
            [self.mdn_mus(x), self.mdn_sigmas(x), self.mdn_pi(x)],
            name="mdn_outputs",
        )


"""
Lets construct an Mixture Density Network using our new layer:
"""

OUTPUT_DIMS = 1
N_MIXES = 20

mdn_network = keras.Sequential(
    [
        layers.Dense(N_HIDDEN, activation="relu"),
        layers.Dense(N_HIDDEN, activation="relu"),
        MixtureDensityOutput(OUTPUT_DIMS, N_MIXES),
    ]
)

"""
Next, let's implement a custom loss function to train the Mixture Density
Network layer based on the true values and our expected outputs:
"""


def get_mixture_loss_func(output_dim, num_mixes):
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistributed layer
        y_pred = ops.reshape(
            y_pred,
            [-1, (2 * num_mixes * output_dim) + num_mixes],
        )
        y_true = ops.reshape(y_true, [-1, output_dim])
        out_mu, out_sigma, out_pi = ops.split(
            y_pred,
            3,
            axis=-1,
        )
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)

        mus = ops.split(out_mu, num_mixes, axis=1)
        sigs = ops.split(out_sigma, num_mixes, axis=1)
        coll = [
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            for loc, scale in zip(mus, sigs)
        ]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = ops.negative(loss)
        loss = ops.mean(loss)
        return loss

    return mdn_loss_func


mdn_network.compile(
    loss=get_mixture_loss_func(OUTPUT_DIMS, N_MIXES), optimizer="adam"
)

"""
Finally, we can call `model.fit()` like any other Keras model.
"""

mdn_network.fit(
    x,
    y,
    epochs=300,
    batch_size=128,
    validation_split=0.15,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="loss", patience=10, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(monitor="loss", patience=5),
    ],
)

"""
Let's make some predictions!
"""

y_pred_mixture = mdn_network.predict(x)

"""
The MDN does not output a single value; instead it outputs values to
parameterize a mixture distribution.
To visualize these outputs, lets sample from the distribution.

Note that sampling is a lossy process.
If you want to preserve all information as part of a greater latent
representation (i.e. for downstream processing) I recommend you simply keep the
distribution parameters in place.
"""


def split_mixture_params(params, output_dim, num_mixes):
    mus = params[: num_mixes * output_dim]
    sigs = params[num_mixes * output_dim : 2 * num_mixes * output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist


def sample_from_categorical(dist):
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    warnings.warn("Error sampling categorical model.")
    return -1


def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim : (m + 1) * output_dim]
    sig_vector = sigs[m * output_dim : (m + 1) * output_dim]
    scale_matrix = (
        np.identity(output_dim) * sig_vector
    )  # scale matrix from diag
    cov_matrix = np.matmul(
        scale_matrix, scale_matrix.T
    )  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample


"""
Next lets use our sampling function:
"""

# Sample from the predicted distributions
y_samples = np.apply_along_axis(
    sample_from_output, 1, y_pred_mixture, 1, N_MIXES, temp=1.0
)

"""
Finally, we can visualize our network outputs
"""

plt.scatter(x, y, alpha=0.05, color="blue", label="Ground Truth")
plt.scatter(
    x,
    y_samples[:, :, 0],
    color="green",
    alpha=0.05,
    label="Mixture Density Network prediction",
)
plt.show()

"""
Beautiful.  Love to see it

# Conclusions

Neural Networks are universal function approximators - but they can only
approximate functions.  Mixture Density networks can approximate arbitrary
x->y mappings using some neat probability tricks.

For more examples with `tensorflow_probability`
[start here](https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression).

One more pretty graphic for the road:
"""

fig, axs = plt.subplots(1, 3)
fig.set_figheight(3)
fig.set_figwidth(12)
axs[0].set_title("Ground Truth")
axs[0].scatter(x, y, alpha=0.05, color="blue")
xlim = axs[0].get_xlim()
ylim = axs[0].get_ylim()

axs[1].set_title("Normal Model prediction")
axs[1].scatter(x, y_pred, alpha=0.05, color="red")
axs[1].set_xlim(xlim)
axs[1].set_ylim(ylim)
axs[2].scatter(
    x,
    y_samples[:, :, 0],
    color="green",
    alpha=0.05,
    label="Mixture Density Network prediction",
)
axs[2].set_title("Mixture Density Network prediction")
axs[2].set_xlim(xlim)
axs[2].set_ylim(ylim)
plt.show()

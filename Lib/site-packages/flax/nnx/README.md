[![codecov](https://codecov.io/gh/cgarciae/nnx/branch/main/graph/badge.svg?token=VqJjL474Z7)](https://codecov.io/gh/cgarciae/nnx)

# NNX

_**N**eural **N**etworks for JA**X**_ - | [docs](https://flax.readthedocs.io/en/latest/nnx/index.html) |

NNX is a JAX-based neural network library that focuses on providing the best development experience to make
building and experimenting with neural networks as easy and intuitive as possible.

* **Pythonic**: Modules are standard Python classes, promoting ease of use and a more familiar
  development experience.
* **Easy-to-use**: NNX provides a set of transforms that take care of state management, allowing
  users to focus on building their models and training loops.
* **Expressive**: NNX allows fine-grained over the Module state with lifted transforms, enabling
  users to define complex architectures.
* **Compatible**: NNX allows functionalizing Module state, making it possible to directly use JAX
  transformations when needed.

## What does NNX look like?

NNX removes most of the friction from building and training neural networks in JAX. It provides
a Module system that uses standard Python classes, and a set of transforms that extend
JAX to handle objects.

```python
from flax import nnx
import optax

class Model(nnx.Module):
  def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dmid, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.dropout = nnx.Dropout(0.2, rngs=rngs)
    self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x):
    x = nnx.relu(self.dropout(self.bn(self.linear(x))))
    return self.linear_out(x)


model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing

@nnx.jit  # automatic state management
def train_step(model, optimizer, x, y):
  def loss_fn(model):
    y_pred = model(x)  # call methods directly
    return ((y_pred - y) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)  # inplace updates

  return loss
```

To learn more about the `Module` abstraction, check out our [NNX Basics](https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html#) guide.

## Installation

To get started with `nnx`, install Flax from GitHub:
```
pip install git+https://github.com/google/flax.git
```

### Examples

* [LM1B](https://github.com/google/flax/tree/main/examples/lm1b_nnx): A language model trained on the 1 Billion Word Benchmark dataset.

#### Toy Examples
* [Basic Example](https://github.com/google/flax/tree/main/examples/nnx_toy_examples/02_lifted_transforms.py): Shows how to train a simple model using NNX.
* [Using the Functional API](https://github.com/google/flax/tree/main/examples/nnx_toy_examples/01_functional_api.py): Shows how to train a simple model using the functional API.
* [Training a VAE](https://github.com/google/flax/tree/main/examples/nnx_toy_examples/05_vae.py): Shows how to train a VAE on the binarized MNIST dataset.
* [Scan over layers](https://github.com/google/flax/tree/main/examples/nnx_toy_examples/06_scan_over_layers.py): An contrived example that implements scan over layers with dropout and a share BatcNorm layer to showcase how lifted transforms can be implemented. It uses the functional API along with `jax.vmap` and `jax.lax.scan`.

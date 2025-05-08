class JaxLayer:
    pass


"""from flax import nnx
import jax.numpy as jnp

class JaxLayer(nnx.Module):
    def __init__(self):
        super().__init__()

    def add_weight(self, name, shape, dtype=None, initializer=None, trainable=True):
        value = initializer(shape, dtype)
        var = nnx.Param(value) if trainable else nnx.Variable(value)
        setattr(self, name, var)
        return var

    def get_weights(self):
        return [v.value for v in nnx.variables(self, nnx.Param)]

    def set_weights(self, weights):
        params = list(nnx.variables(self, nnx.Param).values())
        for var, val in zip(params, weights):
            var.value = val """

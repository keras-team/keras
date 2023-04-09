from keras_core.engine.function import Function
from keras_core.engine.model import Model
from keras_core.engine.layer import Layer


class Functional(Function, Model):
    """
    Add support for extra call arguments compared to Function:
    training, masks

    Add support for arg standardization:
    - list/dict duality
    - upranking

    Override .layers

    Symbolic add_loss
    """

    def __init__(self, inputs, outputs, name=None, **kwargs):
        # This is used by the Model class, since we have some logic to swap the
        # class in the __new__ method, which will lead to __init__ get invoked
        # twice. Using the skip_init to skip one of the invocation of __init__
        # to avoid any side effects
        skip_init = kwargs.pop("skip_init", False)
        if skip_init:
            return
        super().__init__(inputs, outputs, name=name)
        self._layers = self.layers

    @property
    def layers(self):
        layers = []
        for operation in self.operations:
            if isinstance(operation, Layer):
                layers.append(operation)
        return layers

    def call(self, inputs, training=False, mask=None):
        # Add support for traning, masking
        inputs = self._flatten_to_reference_inputs(inputs)
        if mask is None:
            masks = [None] * len(inputs)
        else:
            masks = self._flatten_to_reference_inputs(mask)
        for x, mask in zip(inputs, masks):
            x._keras_mask = mask
        return self._run_through_graph(
            inputs, operation_fn=lambda op: operation_fn(op, training=training)
        )

    def _flatten_to_reference_inputs(self, inputs):
        pass

    def _adjust_input_rank(self, inputs):
        pass

    def _standardize_inputs(self, inputs):
        pass

    def add_loss(self, loss):
        # Symbolic only.
        raise NotImplementedError


def operation_fn(operation, training):
    def call(*arg, **kwargs):
        if operation._call_has_training_arg() and "training" not in kwargs:
            kwargs["training"] = training
        return operation(*arg, **kwargs)

    return call

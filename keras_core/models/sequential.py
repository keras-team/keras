from keras_core.api_export import keras_core_export
from keras_core.models.model import Model


@keras_core_export(["keras_core.Sequential", "keras_core.models.Sequential"])
class Sequential(Model):
    def __init__(self, layers, trainable=True, name=None):
        pass

    def call(self, inputs):
        pass

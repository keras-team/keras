import os

os.environ["KERAS_BACKEND"] = "torch"
import keras_hub

model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
for w in model.weights:
    print(w.name, w.shape)

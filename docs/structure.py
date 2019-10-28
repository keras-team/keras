# -*- coding: utf-8 -*-
'''
General documentation architecture:

Home
Index

- Getting started
    Getting started with the sequential model
    Getting started with the functional api
    FAQ

- Models
    About Keras models
        explain when one should use Sequential or functional API
        explain compilation step
        explain weight saving, weight loading
        explain serialization, deserialization
    Sequential
    Model (functional API)

- Layers
    About Keras layers
        explain common layer functions: get_weights, set_weights, get_config
        explain input_shape
        explain usage on non-Keras tensors
    Core Layers
    Convolutional Layers
    Pooling Layers
    Locally-connected Layers
    Recurrent Layers
    Embedding Layers
    Merge Layers
    Advanced Activations Layers
    Normalization Layers
    Noise Layers
    Layer Wrappers
    Writing your own Keras layers

- Preprocessing
    Sequence Preprocessing
    Text Preprocessing
    Image Preprocessing

Losses
Metrics
Optimizers
Activations
Callbacks
Datasets
Applications
Backend
Initializers
Regularizers
Constraints
Visualization
Scikit-learn API
Utils
Contributing

'''
from keras import utils
from keras import layers
from keras.layers import advanced_activations
from keras.layers import noise
from keras.layers import wrappers
from keras import initializers
from keras import optimizers
from keras import callbacks
from keras import models
from keras import losses
from keras import metrics
from keras import backend
from keras import constraints
from keras import activations
from keras import preprocessing
from keras.preprocessing.image import ImageDataGenerator

from keras_autodoc import get_classes, get_methods, get_functions


save_funcs = {'serialize', 'deserialize'}

# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]
PAGES = {
    'models/sequential.md': [
        models.Sequential.compile,
        models.Sequential.fit,
        models.Sequential.evaluate,
        models.Sequential.predict,
        models.Sequential.train_on_batch,
        models.Sequential.test_on_batch,
        models.Sequential.predict_on_batch,
        models.Sequential.fit_generator,
        models.Sequential.evaluate_generator,
        models.Sequential.predict_generator,
        models.Sequential.get_layer,
    ],
    'models/model.md': [
        models.Model.compile,
        models.Model.fit,
        models.Model.evaluate,
        models.Model.predict,
        models.Model.train_on_batch,
        models.Model.test_on_batch,
        models.Model.predict_on_batch,
        models.Model.fit_generator,
        models.Model.evaluate_generator,
        models.Model.predict_generator,
        models.Model.get_layer,
    ],
    'layers/core.md': [
        layers.Dense,
        layers.Activation,
        layers.Dropout,
        layers.Flatten,
        layers.Reshape,
        layers.Permute,
        layers.RepeatVector,
        layers.Lambda,
        layers.ActivityRegularization,
        layers.Masking,
        layers.SpatialDropout1D,
        layers.SpatialDropout2D,
        layers.SpatialDropout3D,
        layers.Input
    ],
    'layers/convolutional.md': [
        layers.Conv1D,
        layers.Conv2D,
        layers.SeparableConv1D,
        layers.SeparableConv2D,
        layers.DepthwiseConv2D,
        layers.Conv2DTranspose,
        layers.Conv3D,
        layers.Conv3DTranspose,
        layers.Cropping1D,
        layers.Cropping2D,
        layers.Cropping3D,
        layers.UpSampling1D,
        layers.UpSampling2D,
        layers.UpSampling3D,
        layers.ZeroPadding1D,
        layers.ZeroPadding2D,
        layers.ZeroPadding3D,
    ],
    'layers/pooling.md': [
        layers.MaxPooling1D,
        layers.MaxPooling2D,
        layers.MaxPooling3D,
        layers.AveragePooling1D,
        layers.AveragePooling2D,
        layers.AveragePooling3D,
        layers.GlobalMaxPooling1D,
        layers.GlobalAveragePooling1D,
        layers.GlobalMaxPooling2D,
        layers.GlobalAveragePooling2D,
        layers.GlobalMaxPooling3D,
        layers.GlobalAveragePooling3D,
    ],
    'layers/local.md': [
        layers.LocallyConnected1D,
        layers.LocallyConnected2D,
    ],
    'layers/recurrent.md': [
        layers.RNN,
        layers.SimpleRNN,
        layers.GRU,
        layers.LSTM,
        layers.ConvLSTM2D,
        layers.ConvLSTM2DCell,
        layers.SimpleRNNCell,
        layers.GRUCell,
        layers.LSTMCell,
        layers.CuDNNGRU,
        layers.CuDNNLSTM,
    ],
    'layers/embeddings.md': [layers.Embedding],
    'layers/normalization.md': [layers.BatchNormalization],
    'layers/advanced-activations.md': get_classes(advanced_activations,
                                                  exclude=save_funcs),
    'layers/noise.md': get_classes(noise),
    'layers/merge.md': [
        layers.Add,
        layers.Subtract,
        layers.Multiply,
        layers.Average,
        layers.Maximum,
        layers.Minimum,
        layers.Concatenate,
        layers.Dot,
        layers.add,
        layers.subtract,
        layers.multiply,
        layers.average,
        layers.maximum,
        layers.minimum,
        layers.concatenate,
        layers.dot,
    ],
    'preprocessing/sequence.md': [
        preprocessing.sequence.pad_sequences,
        preprocessing.sequence.skipgrams,
        preprocessing.sequence.make_sampling_table,
        preprocessing.sequence.TimeseriesGenerator,
    ],
    'preprocessing/image.md': [ImageDataGenerator]
                              + get_methods(ImageDataGenerator),

    'preprocessing/text.md': [
        preprocessing.text.hashing_trick,
        preprocessing.text.one_hot,
        preprocessing.text.text_to_word_sequence,
        preprocessing.text.Tokenizer,
    ],
    'layers/wrappers.md': get_classes(wrappers, exclude=['Wrapper']),
    'metrics.md': get_functions(metrics),
    'losses.md': get_functions(losses, exclude=save_funcs),
    'initializers.md': get_functions(initializers, exclude=save_funcs)
                       + get_classes(initializers),
    'optimizers.md': get_classes(optimizers),
    'callbacks.md': get_classes(callbacks, exclude=['CallbackList']),
    'activations.md': get_functions(activations),
    'backend.md': get_functions(backend, exclude=['normalize_data_format']),
    'constraints.md': get_classes(constraints),
    'utils.md': [
        utils.to_categorical,
        utils.normalize,
        utils.get_file,
        utils.print_summary,
        utils.plot_model,
        utils.multi_gpu_model,
        utils.CustomObjectScope,
        utils.HDF5Matrix,
        utils.Sequence
    ]
}

ROOT = 'http://keras.io/'

template_np_implementation = """
# Numpy implementation

```python
{{code}}
```
"""

template_hidden_np_implementation = """
# Numpy implementation

<details>
<summary>Show the Numpy implementation</summary>

```python
{{code}}
```

</details>
"""

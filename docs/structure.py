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


EXCLUDE = {
    'Optimizer',
    'TFOptimizer',
    'Wrapper',
    'get_session',
    'set_session',
    'CallbackList',
    'serialize',
    'deserialize',
    'get',
    'set_image_dim_ordering',
    'normalize_data_format',
    'image_dim_ordering',
    'get_variable_shape',
    'Constraint'
}

# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]
PAGES = [
    {
        'page': 'models/sequential.md',
        'methods': [
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
    },
    {
        'page': 'models/model.md',
        'methods': [
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
        ]
    },
    {
        'page': 'layers/core.md',
        'classes': [
            layers.Dense,
            layers.Activation,
            layers.Dropout,
            layers.Flatten,
            layers.Input,
            layers.Reshape,
            layers.Permute,
            layers.RepeatVector,
            layers.Lambda,
            layers.ActivityRegularization,
            layers.Masking,
            layers.SpatialDropout1D,
            layers.SpatialDropout2D,
            layers.SpatialDropout3D,
        ],
    },
    {
        'page': 'layers/convolutional.md',
        'classes': [
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
    },
    {
        'page': 'layers/pooling.md',
        'classes': [
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
    },
    {
        'page': 'layers/local.md',
        'classes': [
            layers.LocallyConnected1D,
            layers.LocallyConnected2D,
        ],
    },
    {
        'page': 'layers/recurrent.md',
        'classes': [
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
    },
    {
        'page': 'layers/embeddings.md',
        'classes': [
            layers.Embedding,
        ],
    },
    {
        'page': 'layers/normalization.md',
        'classes': [
            layers.BatchNormalization,
        ],
    },
    {
        'page': 'layers/advanced-activations.md',
        'all_module_classes': [advanced_activations],
    },
    {
        'page': 'layers/noise.md',
        'all_module_classes': [noise],
    },
    {
        'page': 'layers/merge.md',
        'classes': [
            layers.Add,
            layers.Subtract,
            layers.Multiply,
            layers.Average,
            layers.Maximum,
            layers.Minimum,
            layers.Concatenate,
            layers.Dot,
        ],
        'functions': [
            layers.add,
            layers.subtract,
            layers.multiply,
            layers.average,
            layers.maximum,
            layers.minimum,
            layers.concatenate,
            layers.dot,
        ]
    },
    {
        'page': 'preprocessing/sequence.md',
        'functions': [
            preprocessing.sequence.pad_sequences,
            preprocessing.sequence.skipgrams,
            preprocessing.sequence.make_sampling_table,
        ],
        'classes': [
            preprocessing.sequence.TimeseriesGenerator,
        ]
    },
    {
        'page': 'preprocessing/image.md',
        'classes': [
            (preprocessing.image.ImageDataGenerator, '*')
        ]
    },
    {
        'page': 'preprocessing/text.md',
        'functions': [
            preprocessing.text.hashing_trick,
            preprocessing.text.one_hot,
            preprocessing.text.text_to_word_sequence,
        ],
        'classes': [
            preprocessing.text.Tokenizer,
        ]
    },
    {
        'page': 'layers/wrappers.md',
        'all_module_classes': [wrappers],
    },
    {
        'page': 'metrics.md',
        'all_module_functions': [metrics],
    },
    {
        'page': 'losses.md',
        'all_module_functions': [losses],
    },
    {
        'page': 'initializers.md',
        'all_module_functions': [initializers],
        'all_module_classes': [initializers],
    },
    {
        'page': 'optimizers.md',
        'all_module_classes': [optimizers],
    },
    {
        'page': 'callbacks.md',
        'all_module_classes': [callbacks],
    },
    {
        'page': 'activations.md',
        'all_module_functions': [activations],
    },
    {
        'page': 'backend.md',
        'all_module_functions': [backend],
    },
    {
        'page': 'constraints.md',
        'all_module_classes': [constraints],
    },
    {
        'page': 'utils.md',
        'functions': [utils.to_categorical,
                      utils.normalize,
                      utils.get_file,
                      utils.print_summary,
                      utils.plot_model,
                      utils.multi_gpu_model],
        'classes': [utils.CustomObjectScope,
                    utils.HDF5Matrix,
                    utils.Sequence],
    },
]

ROOT = 'http://keras.io/'

template_np_implementation = """# Numpy implementation

    ```python
{{code}}
    ```
"""

template_hidden_np_implementation = """# Numpy implementation

    <details>
    <summary>Show the Numpy implementation</summary>

    ```python
{{code}}
    ```

    </details>
"""

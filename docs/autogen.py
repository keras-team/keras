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
from __future__ import print_function
from __future__ import unicode_literals

import re
import inspect
import os
import shutil

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
from keras import activations

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding('utf8')


EXCLUDE = {
    'Optimizer',
    'Wrapper',
    'get_session',
    'set_session',
    'CallbackList',
    'serialize',
    'deserialize',
    'get',
    'set_image_dim_ordering',
    'image_dim_ordering',
    'get_variable_shape',
}

PAGES = [
    {
        'page': 'models/sequential.md',
        'functions': [
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
        'functions': [
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
        ],
    },
    {
        'page': 'layers/convolutional.md',
        'classes': [
            layers.Conv1D,
            layers.Conv2D,
            layers.SeparableConv2D,
            layers.Conv2DTranspose,
            layers.Conv3D,
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
            layers.SimpleRNNCell,
            layers.GRUCell,
            layers.LSTMCell,
            layers.StackedRNNCells,
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
            layers.Concatenate,
            layers.Dot,
        ],
        'functions': [
            layers.add,
            layers.subtract,
            layers.multiply,
            layers.average,
            layers.maximum,
            layers.concatenate,
            layers.dot,
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


def get_earliest_class_that_defined_member(member, cls):
    ancestors = get_classes_ancestors([cls])
    result = None
    for ancestor in ancestors:
        if member in dir(ancestor):
            result = ancestor
    if not result:
        return cls
    return result


def get_classes_ancestors(classes):
    ancestors = []
    for cls in classes:
        ancestors += cls.__bases__
    filtered_ancestors = []
    for ancestor in ancestors:
        if ancestor.__name__ in ['object']:
            continue
        filtered_ancestors.append(ancestor)
    if filtered_ancestors:
        return filtered_ancestors + get_classes_ancestors(filtered_ancestors)
    else:
        return filtered_ancestors


def get_function_signature(function, method=True):
    wrapped = getattr(function, '_original_function', None)
    if wrapped is None:
        signature = inspect.getargspec(function)
    else:
        signature = inspect.getargspec(wrapped)
    defaults = signature.defaults
    if method:
        args = signature.args[1:]
    else:
        args = signature.args
    if defaults:
        kwargs = zip(args[-len(defaults):], defaults)
        args = args[:-len(defaults)]
    else:
        kwargs = []
    st = '%s.%s(' % (function.__module__, function.__name__)

    for a in args:
        st += str(a) + ', '
    for a, v in kwargs:
        if isinstance(v, str):
            v = '\'' + v + '\''
        st += str(a) + '=' + str(v) + ', '
    if kwargs or args:
        signature = st[:-2] + ')'
    else:
        signature = st + ')'

    if not method:
        # Prepend the module name.
        signature = function.__module__ + '.' + signature
    return post_process_signature(signature)


def get_class_signature(cls):
    try:
        class_signature = get_function_signature(cls.__init__)
        class_signature = class_signature.replace('__init__', cls.__name__)
    except (TypeError, AttributeError):
        # in case the class inherits from object and does not
        # define __init__
        class_signature = cls.__module__ + '.' + cls.__name__ + '()'
    return post_process_signature(class_signature)


def post_process_signature(signature):
    parts = re.split('\.(?!\d)', signature)
    if len(parts) >= 4:
        if parts[1] == 'layers':
            signature = 'keras.layers.' + '.'.join(parts[3:])
        if parts[1] == 'utils':
            signature = 'keras.utils.' + '.'.join(parts[3:])
        if parts[1] == 'backend':
            signature = 'keras.backend.' + '.'.join(parts[3:])
    return signature


def class_to_docs_link(cls):
    module_name = cls.__module__
    assert module_name[:6] == 'keras.'
    module_name = module_name[6:]
    link = ROOT + module_name.replace('.', '/') + '#' + cls.__name__.lower()
    return link


def class_to_source_link(cls):
    module_name = cls.__module__
    assert module_name[:6] == 'keras.'
    path = module_name.replace('.', '/')
    path += '.py'
    line = inspect.getsourcelines(cls)[-1]
    link = ('https://github.com/keras-team/'
            'keras/blob/master/' + path + '#L' + str(line))
    return '[[source]](' + link + ')'


def code_snippet(snippet):
    result = '```python\n'
    result += snippet + '\n'
    result += '```\n'
    return result


def count_leading_spaces(s):
    ws = re.search('\S', s)
    if ws:
        return ws.start()
    else:
        return 0


def process_docstring(docstring):
    # First, extract code blocks and process them.
    code_blocks = []
    if '```' in docstring:
        tmp = docstring[:]
        while '```' in tmp:
            tmp = tmp[tmp.find('```'):]
            index = tmp[3:].find('```') + 6
            snippet = tmp[:index]
            # Place marker in docstring for later reinjection.
            docstring = docstring.replace(
                snippet, '$CODE_BLOCK_%d' % len(code_blocks))
            snippet_lines = snippet.split('\n')
            # Remove leading spaces.
            num_leading_spaces = snippet_lines[-1].find('`')
            snippet_lines = ([snippet_lines[0]] +
                             [line[num_leading_spaces:]
                             for line in snippet_lines[1:]])
            # Most code snippets have 3 or 4 more leading spaces
            # on inner lines, but not all. Remove them.
            inner_lines = snippet_lines[1:-1]
            leading_spaces = None
            for line in inner_lines:
                if not line or line[0] == '\n':
                    continue
                spaces = count_leading_spaces(line)
                if leading_spaces is None:
                    leading_spaces = spaces
                if spaces < leading_spaces:
                    leading_spaces = spaces
            if leading_spaces:
                snippet_lines = ([snippet_lines[0]] +
                                 [line[leading_spaces:]
                                  for line in snippet_lines[1:-1]] +
                                 [snippet_lines[-1]])
            snippet = '\n'.join(snippet_lines)
            code_blocks.append(snippet)
            tmp = tmp[index:]

    # Format docstring section titles.
    docstring = re.sub(r'\n(\s+)# (.*)\n',
                       r'\n\1__\2__\n\n',
                       docstring)
    # Format docstring lists.
    docstring = re.sub(r'    ([^\s\\\(]+):(.*)\n',
                       r'    - __\1__:\2\n',
                       docstring)

    # Strip all leading spaces.
    lines = docstring.split('\n')
    docstring = '\n'.join([line.lstrip(' ') for line in lines])

    # Reinject code blocks.
    for i, code_block in enumerate(code_blocks):
        docstring = docstring.replace(
            '$CODE_BLOCK_%d' % i, code_block)
    return docstring

print('Cleaning up existing sources directory.')
if os.path.exists('sources'):
    shutil.rmtree('sources')

print('Populating sources directory with templates.')
for subdir, dirs, fnames in os.walk('templates'):
    for fname in fnames:
        new_subdir = subdir.replace('templates', 'sources')
        if not os.path.exists(new_subdir):
            os.makedirs(new_subdir)
        if fname[-3:] == '.md':
            fpath = os.path.join(subdir, fname)
            new_fpath = fpath.replace('templates', 'sources')
            shutil.copy(fpath, new_fpath)


# Take care of index page.
def read_file(path):
    with open(path) as f:
        return f.read()


readme = read_file('../README.md')
index = read_file('templates/index.md')
index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
with open('sources/index.md', 'w') as f:
    f.write(index)

print('Starting autogeneration.')
for page_data in PAGES:
    blocks = []
    classes = page_data.get('classes', [])
    for module in page_data.get('all_module_classes', []):
        module_classes = []
        for name in dir(module):
            if name[0] == '_' or name in EXCLUDE:
                continue
            module_member = getattr(module, name)
            if inspect.isclass(module_member):
                cls = module_member
                if cls.__module__ == module.__name__:
                    if cls not in module_classes:
                        module_classes.append(cls)
        module_classes.sort(key=lambda x: id(x))
        classes += module_classes

    for cls in classes:
        subblocks = []
        signature = get_class_signature(cls)
        subblocks.append('<span style="float:right;">' +
                         class_to_source_link(cls) + '</span>')
        subblocks.append('### ' + cls.__name__ + '\n')
        subblocks.append(code_snippet(signature))
        docstring = cls.__doc__
        if docstring:
            subblocks.append(process_docstring(docstring))
        blocks.append('\n'.join(subblocks))

    functions = page_data.get('functions', [])
    for module in page_data.get('all_module_functions', []):
        module_functions = []
        for name in dir(module):
            if name[0] == '_' or name in EXCLUDE:
                continue
            module_member = getattr(module, name)
            if inspect.isfunction(module_member):
                function = module_member
                if module.__name__ in function.__module__:
                    if function not in module_functions:
                        module_functions.append(function)
        module_functions.sort(key=lambda x: id(x))
        functions += module_functions

    for function in functions:
        subblocks = []
        signature = get_function_signature(function, method=False)
        signature = signature.replace(function.__module__ + '.', '')
        subblocks.append('### ' + function.__name__ + '\n')
        subblocks.append(code_snippet(signature))
        docstring = function.__doc__
        if docstring:
            subblocks.append(process_docstring(docstring))
        blocks.append('\n\n'.join(subblocks))

    if not blocks:
        raise RuntimeError('Found no content for page ' +
                           page_data['page'])

    mkdown = '\n----\n\n'.join(blocks)
    # save module page.
    # Either insert content into existing page,
    # or create page otherwise
    page_name = page_data['page']
    path = os.path.join('sources', page_name)
    if os.path.exists(path):
        template = read_file(path)
        assert '{{autogenerated}}' in template, ('Template found for ' + path +
                                                 ' but missing {{autogenerated}} tag.')
        mkdown = template.replace('{{autogenerated}}', mkdown)
        print('...inserting autogenerated content into template:', path)
    else:
        print('...creating new page with autogenerated content:', path)
    subdir = os.path.dirname(path)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    with open(path, 'w') as f:
        f.write(mkdown)

shutil.copyfile('../CONTRIBUTING.md', 'sources/contributing.md')

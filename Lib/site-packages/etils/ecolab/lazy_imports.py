# Copyright 2024 The etils Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common lazy imports.

Usage:

```python
from etils.ecolab.lazy_imports import *
```

To get the list of available modules:

```python
lazy_imports.__all__  # List of modules aliases
lazy_imports.LAZY_MODULES  # Mapping <module_alias>: <lazy_module info>
```
"""

from __future__ import annotations

from etils.ecolab import lazy_utils


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  """`lazy_imports` public API.

  Because `globals()` contains hundreds of symbols, we overwrite `dir(module)`
  to avoid poluting the namespace during auto-completion.

  Returns:
    public symbols
  """
  # If modifying this, also update the `lazy_imports/__init__.py``
  return [
      '__all__',
      'LAZY_MODULES',
      'print_current_imports',
  ]


def print_current_imports() -> None:
  """Display the active lazy imports.

  This can be used before publishing a colab. To convert lazy imports
  into explicit imports.

  For convenience, `from etils.ecolab import lazy_imports` is excluded from
  the current imports.
  """
  print(lazy_utils.current_import_statements(LAZY_MODULES))


_builder = lazy_utils.LazyImportsBuilder(globals())


with _builder.replace_imports(is_std=True):
  # pylint: disable=g-import-not-at-top,unused-import,reimported
  import abc
  import argparse
  import ast
  import asyncio
  import base64
  import builtins
  import collections
  import colorsys
  import copy
  import concurrent.futures
  import contextlib
  import contextvars
  import csv
  import dataclasses
  import datetime
  import difflib
  import dis
  import enum
  import functools
  import gc
  import gzip
  import html
  import inspect
  import io
  import importlib
  import IPython
  import itertools
  import json
  import logging
  import math
  import multiprocessing
  import os
  import pathlib
  import pdb
  import pickle
  import pprint
  import queue
  import random
  import re
  import shutil
  import stat
  import string
  import subprocess
  import sys
  import tarfile
  import textwrap
  import threading
  import time
  import timeit
  import tomllib  # pytype: disable=import-error
  import traceback
  import typing  # Note we do not import `Any`, `TypeVar`,...
  import types
  import urllib
  import uuid
  from unittest import mock
  import warnings
  import weakref
  import zipfile
  # pylint: enable=g-import-not-at-top,unused-import,reimported


with _builder.replace_imports(is_std=False):
  # pylint: disable=g-import-not-at-top,unused-import,reimported
  # pytype: disable=import-error
  # ====== Etils ======
  from etils import array_types
  from etils import ecolab
  from etils import edc
  from etils import enp
  from etils import epath
  from etils import epy
  from etils import etqdm
  from etils import etree
  from etils import exm
  from etils import g3_utils
  from etils.ecolab import lazy_imports
  # ====== Common third party ======
  from absl import app
  from absl import flags
  import apache_beam as beam
  import chex
  import dataclass_array as dca
  import einops
  import flask
  import flax
  from flax import linen as nn
  from flax import nnx
  import functorch
  import gin
  import grain.python as grain
  import graphviz
  import imageio
  # Even though `import ipywidgets as widgets` is the common alias, widgets
  # is likely too ambiguous.
  import ipywidgets
  import jax
  from jax import numpy as jnp
  import jaxtyping
  import lark
  import matplotlib
  import matplotlib as mpl  # Standard alias
  from matplotlib import pyplot as plt
  import mediapy as media
  import ml_collections
  import networkx as nx
  import numpy as np
  import optax
  import orbax
  from orbax import checkpoint as ocp
  from orbax import export as oex
  import pandas as pd
  import PIL
  from PIL import Image  # Common alias
  import pycolmap
  import scipy
  import seaborn as sns
  import sklearn
  import tensorflow as tf
  import tensorflow.experimental.numpy as tnp
  import tensorflow_datasets as tfds
  import torch
  # from torch import nn  # Collision with flax.linen
  import torchtext
  import torchvision
  import tqdm
  # tqdm import also trigger additional imports.
  # TODO(epot): Currently pylance might not infer `tqdm.auto` match
  # `import tqdm.auto`
  # Could try to explicitly import inside a `if typing.TYPE_CHECKING:`
  tqdm.auto  # pylint: disable=pointless-statement
  tqdm.notebook  # pylint: disable=pointless-statement
  import tree
  import typeguard
  import typing_extensions
  import plotly
  from plotly import express as px
  from plotly import graph_objects as go
  from pydantic import v1 as pydantic
  import requests
  import sunds
  import visu3d as v3d
  from xmanager.contrib import flow as xmflow
  from xmanager import xm
  # pytype: enable=import-error
  # pylint: enable=g-import-not-at-top,unused-import,reimported


# Sort the lazy modules per their <module_name>
LAZY_MODULES: dict[str, lazy_utils.LazyModule] = dict(
    sorted(
        _builder.lazy_modules.items(),
        key=lambda x: x[1]._etils_state.module_name,  # pylint: disable=protected-access
    )
)

__all__ = sorted(LAZY_MODULES)  # Sorted per alias

from pyreadline3.py3k_compat import is_ironpython

from . import winconstants

if is_ironpython:
    try:
        from .ironpython_keysyms import *
    except ImportError as x:
        raise ImportError("Could not import keysym for local ironpython version") from x
else:
    try:
        from .keysyms import *
    except ImportError as x:
        raise ImportError("Could not import keysym for local python version") from x

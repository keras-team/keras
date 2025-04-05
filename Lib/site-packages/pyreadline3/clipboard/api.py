# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from pyreadline3.py3k_compat import is_ironpython

if is_ironpython:
    try:
        from .ironpython_clipboard import get_clipboard_text, set_clipboard_text
    except ImportError:
        from .no_clipboard import get_clipboard_text, set_clipboard_text

else:
    try:
        from .win32_clipboard import get_clipboard_text, set_clipboard_text
    except ImportError:
        from .no_clipboard import get_clipboard_text, set_clipboard_text

__all__ = [
    "get_clipboard_text",
    "set_clipboard_text",
]

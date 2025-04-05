# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from .api import set_clipboard_text
from .get_clipboard_text_and_convert import get_clipboard_text_and_convert

__all__ = [
    "set_clipboard_text",
    "get_clipboard_text_and_convert",
]

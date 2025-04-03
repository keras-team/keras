# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************
import sys
from typing import Iterable, Union

#: Also support non-latin chars.
_pyreadline_fallback_codepage = "utf-8"
try:
    pyreadline_codepage = sys.stdout.encoding
except AttributeError:
    # This error occurs when pdb imports readline and doctest has replaced
    # stdout with stdout collector. We will assume ascii codepage
    pyreadline_codepage = _pyreadline_fallback_codepage

if pyreadline_codepage is None:
    pyreadline_codepage = _pyreadline_fallback_codepage


def ensure_unicode(text: Union[str, bytes]) -> str:
    """helper to ensure that text passed to WriteConsoleW is unicode"""

    if isinstance(text, bytes):
        try:
            return text.decode(pyreadline_codepage, "replace")
        except (LookupError, TypeError):
            return text.decode(errors="replace")

    return text


def ensure_str(text: Union[str, bytes]) -> bytes:
    """Convert unicode to str using pyreadline_codepage"""

    if isinstance(text, str):
        try:
            return text.encode(pyreadline_codepage, "replace")
        except (LookupError, TypeError):
            return text.encode(errors="replace")

    return text


def biter(text: bytes) -> Iterable[bytes]:
    if isinstance(text, bytes):
        return (s.to_bytes(1, "big") for s in text)

    return iter(text)

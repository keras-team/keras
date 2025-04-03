# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from typing import Any, List, Tuple

from .api import get_clipboard_text


def _make_num(x: Any) -> Any:
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            try:
                return complex(x)
            except ValueError:
                return x


def _make_list_of_list(txt: str) -> Tuple[
    List[List[Any]],
    bool,
]:

    ut = []
    flag = False

    for line in [x for x in txt.split("\r\n") if x != ""]:
        words = [_make_num(x) for x in line.split("\t")]

        if str in list(map(type, words)):
            flag = True

        ut.append(words)

    return (
        ut,
        flag,
    )


def get_clipboard_text_and_convert(paste_list: bool = False) -> str:
    """Get txt from clipboard. if paste_list==True the convert tab separated
    data to list of lists. Enclose list of list in array() if all elements are
    numeric"""

    txt = get_clipboard_text()

    if not txt:
        return ""

    if not paste_list:
        return txt

    if "\t" not in txt:
        return txt

    array, flag = _make_list_of_list(txt)

    if flag:
        txt = repr(array)
    else:
        txt = "array(%s)" % repr(array)

    txt = "".join([c for c in txt if c not in " \t\r\n"])

    return txt

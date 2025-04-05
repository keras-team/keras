# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

from typing import Any, List

from .api import set_clipboard_text


def _make_tab(lists: List[Any]) -> str:
    if hasattr(lists, "tolist"):
        lists = lists.tolist()

    ut = []
    for rad in lists:
        if type(rad) in [list, tuple]:
            ut.append("\t".join(["%s" % x for x in rad]))
        else:
            ut.append("%s" % rad)

    return "\n".join(ut)


def _send_data(lists: List[Any]) -> None:
    set_clipboard_text(_make_tab(lists))

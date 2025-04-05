# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import clr
import System.Windows.Forms.Clipboard as cb

clr.AddReferenceByPartialName("System.Windows.Forms")


def get_clipboard_text() -> str:
    text = ""
    if cb.ContainsText():
        text = cb.GetText()

    return text


def set_clipboard_text(text: str) -> None:
    cb.SetText(text)


if __name__ == "__main__":
    txt = get_clipboard_text()  # display last text clipped
    print(txt)

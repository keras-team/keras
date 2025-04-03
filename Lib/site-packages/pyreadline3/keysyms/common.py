# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************
# table for translating virtual keys to X windows key symbols


try:
    set
except NameError:
    from sets import Set as set

from pyreadline3.unicode_helper import ensure_unicode

validkey = set(
    [
        "cancel",
        "backspace",
        "tab",
        "clear",
        "return",
        "shift_l",
        "control_l",
        "alt_l",
        "pause",
        "caps_lock",
        "escape",
        "space",
        "prior",
        "next",
        "end",
        "home",
        "left",
        "up",
        "right",
        "down",
        "select",
        "print",
        "execute",
        "snapshot",
        "insert",
        "delete",
        "help",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
        "f13",
        "f14",
        "f15",
        "f16",
        "f17",
        "f18",
        "f19",
        "f20",
        "f21",
        "f22",
        "f23",
        "f24",
        "num_lock",
        "scroll_lock",
        "vk_apps",
        "vk_processkey",
        "vk_attn",
        "vk_crsel",
        "vk_exsel",
        "vk_ereof",
        "vk_play",
        "vk_zoom",
        "vk_noname",
        "vk_pa1",
        "vk_oem_clear",
        "numpad0",
        "numpad1",
        "numpad2",
        "numpad3",
        "numpad4",
        "numpad5",
        "numpad6",
        "numpad7",
        "numpad8",
        "numpad9",
        "divide",
        "multiply",
        "add",
        "subtract",
        "vk_decimal",
    ]
)

escape_sequence_to_special_key = {
    "\\e[a": "up",
    "\\e[b": "down",
    "del": "delete",
}


class KeyPress(object):
    def __init__(self, char="", shift=False, control=False, meta=False, keyname=""):
        if control or meta or shift:
            char = char.upper()
        self.info = dict(
            char=char, shift=shift, control=control, meta=meta, keyname=keyname
        )

    def create(name):
        def get(self):
            return self.info[name]

        def set(self, value):
            self.info[name] = value

        return property(get, set)

    char = create("char")
    shift = create("shift")
    control = create("control")
    meta = create("meta")
    keyname = create("keyname")

    def __repr__(self):
        return "(%s,%s,%s,%s)" % tuple(map(ensure_unicode, self.tuple()))

    def tuple(self):
        if self.keyname:
            return (self.control, self.meta, self.shift, self.keyname)
        else:
            if self.control or self.meta or self.shift:
                return (self.control, self.meta, self.shift, self.char.upper())
            else:
                return (self.control, self.meta, self.shift, self.char)

    def __eq__(self, other):
        if isinstance(other, KeyPress):
            s = self.tuple()
            o = other.tuple()
            return s == o
        else:
            return False


def make_KeyPress_from_keydescr(keydescr):
    keyinfo = KeyPress()
    if len(keydescr) > 2 and keydescr[:1] == '"' and keydescr[-1:] == '"':
        keydescr = keydescr[1:-1]

    while True:
        lkeyname = keydescr.lower()
        if lkeyname.startswith("control-"):
            keyinfo.control = True
            keydescr = keydescr[8:]
        elif lkeyname.startswith("ctrl-"):
            keyinfo.control = True
            keydescr = keydescr[5:]
        elif keydescr.lower().startswith("\\c-"):
            keyinfo.control = True
            keydescr = keydescr[3:]
        elif keydescr.lower().startswith("\\m-"):
            keyinfo.meta = True
            keydescr = keydescr[3:]
        elif keydescr in escape_sequence_to_special_key:
            keydescr = escape_sequence_to_special_key[keydescr]
        elif lkeyname.startswith("meta-"):
            keyinfo.meta = True
            keydescr = keydescr[5:]
        elif lkeyname.startswith("alt-"):
            keyinfo.meta = True
            keydescr = keydescr[4:]
        elif lkeyname.startswith("shift-"):
            keyinfo.shift = True
            keydescr = keydescr[6:]
        else:
            if len(keydescr) > 1:
                if keydescr.strip().lower() in validkey:
                    keyinfo.keyname = keydescr.strip().lower()
                    keyinfo.char = ""
                else:
                    raise IndexError("Not a valid key: '%s'" % keydescr)
            else:
                keyinfo.char = keydescr
            return keyinfo


if __name__ == "__main__":
    import startup

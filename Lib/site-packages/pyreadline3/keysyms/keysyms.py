# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


from ctypes import windll

from . import winconstants as c32
from .common import KeyPress

# table for translating virtual keys to X windows key symbols

code2sym_map = {
    c32.VK_CANCEL: "cancel",
    c32.VK_BACK: "backspace",
    c32.VK_TAB: "tab",
    c32.VK_CLEAR: "clear",
    c32.VK_RETURN: "return",
    c32.VK_SHIFT: "shift_l",
    c32.VK_CONTROL: "control_l",
    c32.VK_MENU: "alt_l",
    c32.VK_PAUSE: "pause",
    c32.VK_CAPITAL: "caps_lock",
    c32.VK_ESCAPE: "escape",
    c32.VK_SPACE: "space",
    c32.VK_PRIOR: "prior",
    c32.VK_NEXT: "next",
    c32.VK_END: "end",
    c32.VK_HOME: "home",
    c32.VK_LEFT: "left",
    c32.VK_UP: "up",
    c32.VK_RIGHT: "right",
    c32.VK_DOWN: "down",
    c32.VK_SELECT: "select",
    c32.VK_PRINT: "print",
    c32.VK_EXECUTE: "execute",
    c32.VK_SNAPSHOT: "snapshot",
    c32.VK_INSERT: "insert",
    c32.VK_DELETE: "delete",
    c32.VK_HELP: "help",
    c32.VK_F1: "f1",
    c32.VK_F2: "f2",
    c32.VK_F3: "f3",
    c32.VK_F4: "f4",
    c32.VK_F5: "f5",
    c32.VK_F6: "f6",
    c32.VK_F7: "f7",
    c32.VK_F8: "f8",
    c32.VK_F9: "f9",
    c32.VK_F10: "f10",
    c32.VK_F11: "f11",
    c32.VK_F12: "f12",
    c32.VK_F13: "f13",
    c32.VK_F14: "f14",
    c32.VK_F15: "f15",
    c32.VK_F16: "f16",
    c32.VK_F17: "f17",
    c32.VK_F18: "f18",
    c32.VK_F19: "f19",
    c32.VK_F20: "f20",
    c32.VK_F21: "f21",
    c32.VK_F22: "f22",
    c32.VK_F23: "f23",
    c32.VK_F24: "f24",
    c32.VK_NUMLOCK: "num_lock,",
    c32.VK_SCROLL: "scroll_lock",
    c32.VK_APPS: "vk_apps",
    c32.VK_PROCESSKEY: "vk_processkey",
    c32.VK_ATTN: "vk_attn",
    c32.VK_CRSEL: "vk_crsel",
    c32.VK_EXSEL: "vk_exsel",
    c32.VK_EREOF: "vk_ereof",
    c32.VK_PLAY: "vk_play",
    c32.VK_ZOOM: "vk_zoom",
    c32.VK_NONAME: "vk_noname",
    c32.VK_PA1: "vk_pa1",
    c32.VK_OEM_CLEAR: "vk_oem_clear",
    c32.VK_NUMPAD0: "numpad0",
    c32.VK_NUMPAD1: "numpad1",
    c32.VK_NUMPAD2: "numpad2",
    c32.VK_NUMPAD3: "numpad3",
    c32.VK_NUMPAD4: "numpad4",
    c32.VK_NUMPAD5: "numpad5",
    c32.VK_NUMPAD6: "numpad6",
    c32.VK_NUMPAD7: "numpad7",
    c32.VK_NUMPAD8: "numpad8",
    c32.VK_NUMPAD9: "numpad9",
    c32.VK_DIVIDE: "divide",
    c32.VK_MULTIPLY: "multiply",
    c32.VK_ADD: "add",
    c32.VK_SUBTRACT: "subtract",
    c32.VK_DECIMAL: "vk_decimal",
}

VkKeyScan = windll.user32.VkKeyScanA


def char_to_keyinfo(char, control=False, meta=False, shift=False):
    k = KeyPress()
    vk = VkKeyScan(ord(char))
    if vk & 0xFFFF == 0xFFFF:
        print('VkKeyScan("%s") = %x' % (char, vk))
        raise ValueError("bad key")
    if vk & 0x100:
        k.shift = True
    if vk & 0x200:
        k.control = True
    if vk & 0x400:
        k.meta = True
    k.char = chr(vk & 0xFF)
    return k


def make_KeyPress(char, state, keycode):
    control = (state & (4 + 8)) != 0
    meta = (state & (1 + 2)) != 0
    shift = (state & 0x10) != 0
    if control and not meta:  # Matches ctrl- chords should pass keycode as char
        char = chr(keycode)
    elif control and meta:  # Matches alt gr and should just pass on char
        control = False
        meta = False
    try:
        keyname = code2sym_map[keycode]
    except KeyError:
        keyname = ""
    out = KeyPress(char, shift, control, meta, keyname)
    return out


if __name__ == "__main__":
    import startup

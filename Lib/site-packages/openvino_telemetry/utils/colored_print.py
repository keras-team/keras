# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from platform import system


def colored_print(text: str):
    platform = system()
    if platform == 'Windows':
        from ctypes import windll, Structure, byref
        from ctypes import wintypes

        class CONSOLE_SCREEN_BUFFER_INFO(Structure):
            _fields_ = [
                ("dwSize", wintypes._COORD),
                ("dwCursorPosition", wintypes._COORD),
                ("wAttributes", wintypes.WORD),
                ("srWindow", wintypes.SMALL_RECT),
                ("dwMaximumWindowSize", wintypes._COORD),
            ]

        console_screen_buffer_info = CONSOLE_SCREEN_BUFFER_INFO()
        windll.kernel32.GetConsoleScreenBufferInfo(windll.kernel32.GetStdHandle(-11), byref(console_screen_buffer_info))
        windll.kernel32.SetConsoleTextAttribute(windll.kernel32.GetStdHandle(-11), 0x0002)
        print(text)
        windll.kernel32.SetConsoleTextAttribute(windll.kernel32.GetStdHandle(-11),
                                                console_screen_buffer_info.wAttributes)
    else:
        print('\033[32m' + text + '\033[0m')

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from platform import system


def print_without_end_line(string: str):
    """
    Prints a string without end line.
    :param string: a string that will be printed on the screen.
    """
    sys.stdout.write(string)
    sys.stdout.flush()


def input_with_timeout(prompt: str, timeout: int):
    """
    Requests user input and waits until the timeout expired.
    :param prompt: a string that will be printed on the screen.
    :param timeout: timeout to wait.
    :return: input string or empty line if the timeout expired.
    """
    platform = system()
    res_str = ''
    if platform == 'Windows':
        import msvcrt
        import time

        print_without_end_line(prompt)
        start_time = time.monotonic()
        end_time = start_time + timeout
        sleep_time = 0.05

        while time.monotonic() < end_time:
            if msvcrt.kbhit():
                c = msvcrt.getwche()
                if c in ['\r', '\n']:
                    print()
                    return res_str
                if c == '\003':
                    raise KeyboardInterrupt
                if c == '\b':
                    res_str = res_str[:-1]
                    print_without_end_line(''.join(['\r', ' ' * len(prompt + res_str + ' '), '\r', prompt, res_str]))
                else:
                    res_str += c
            time.sleep(sleep_time)

        print()
        return ''
    else:
        import selectors

        print_without_end_line(prompt)
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin, selectors.EVENT_READ)
            events = selector.select(timeout)

            if events:
                key, _ = events[0]
                res_str = key.fileobj.readline().rstrip('\n')
            else:
                print()
            selector.unregister(sys.stdin)
        return res_str

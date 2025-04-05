# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import re


def str_find_all(in_str, ch):
    result = []
    index = 0
    while index >= 0:
        index = in_str.find(ch, index)
        if index >= 0:
            result.append(index)
            index += 1
    return result


word_pattern = re.compile("(x*)")


def markwords(in_str, is_wordfun):
    markers = {True: "x", False: "o"}
    return "".join([markers[is_wordfun(ch)] for ch in in_str])


def split_words(in_str, is_wordfun):
    return [x for x in word_pattern.split(markwords(in_str, is_wordfun)) if x != ""]


def mark_start_segment(in_str, is_segment):
    def mark_start(s):
        if s[0:1] == "x":
            return "s" + s[1:]
        else:
            return s

    return "".join(map(mark_start, split_words(in_str, is_segment)))


def mark_end_segment(in_str, is_segment):
    def mark_start(s):
        if s[0:1] == "x":
            return s[:-1] + "s"
        else:
            return s

    return "".join(map(mark_start, split_words(in_str, is_segment)))


def mark_start_segment_index(in_str, is_segment):
    return str_find_all(mark_start_segment(in_str, is_segment), "s")


def mark_end_segment_index(in_str, is_segment):
    return [x + 1 for x in str_find_all(mark_end_segment(in_str, is_segment), "s")]


# ###############  Following are used in lineobj  ###########################


def is_word_token(in_str):
    return not is_non_word_token(in_str)


def is_non_word_token(in_str):
    if len(in_str) != 1 or in_str in " \t\n":
        return True
    else:
        return False


def next_start_segment(in_str, is_segment):
    in_str = "".join(in_str)
    result = []
    for start in mark_start_segment_index(in_str, is_segment):
        result[len(result) : start] = [start for x in range(start - len(result))]
    result[len(result) : len(in_str)] = [
        len(in_str) for x in range(len(in_str) - len(result) + 1)
    ]
    return result


def next_end_segment(in_str, is_segment):
    in_str = "".join(in_str)
    result = []
    for start in mark_end_segment_index(in_str, is_segment):
        result[len(result) : start] = [start for x in range(start - len(result))]
    result[len(result) : len(in_str)] = [
        len(in_str) for x in range(len(in_str) - len(result) + 1)
    ]
    return result


def prev_start_segment(in_str, is_segment):
    in_str = "".join(in_str)
    result = []
    prev = 0
    for start in mark_start_segment_index(in_str, is_segment):
        result[len(result) : start + 1] = [prev for x in range(start - len(result) + 1)]
        prev = start
    result[len(result) : len(in_str)] = [
        prev for x in range(len(in_str) - len(result) + 1)
    ]
    return result


def prev_end_segment(in_str, is_segment):
    in_str = "".join(in_str)
    result = []
    prev = 0
    for start in mark_end_segment_index(in_str, is_segment):
        result[len(result) : start + 1] = [prev for x in range(start - len(result) + 1)]
        prev = start
    result[len(result) : len(in_str)] = [
        len(in_str) for x in range(len(in_str) - len(result) + 1)
    ]
    return result

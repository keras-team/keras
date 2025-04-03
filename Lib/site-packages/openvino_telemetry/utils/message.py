# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class MessageType(Enum):
    EVENT = 0
    ERROR = 1
    STACK_TRACE = 2
    SESSION_START = 3
    SESSION_END = 4


class Message:
    def __init__(self, type: MessageType, attrs: dict):
        self.type = type
        self.attrs = attrs.copy()

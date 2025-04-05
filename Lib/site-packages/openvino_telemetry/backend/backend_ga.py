# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import uuid
from urllib import request, parse

from .backend import TelemetryBackend
from ..utils.cid import get_or_generate_cid, remove_cid_file
from ..utils.message import Message, MessageType


class GABackend(TelemetryBackend):
    backend_url = 'https://www.google-analytics.com/collect'
    id = 'ga'
    cid_filename = 'openvino_ga_cid'

    def __init__(self, tid: str = None, app_name: str = None, app_version: str = None):
        super(GABackend, self).__init__(tid, app_name, app_version)
        self.tid = tid
        self.cid = None
        self.app_name = app_name
        self.app_version = app_version
        self.default_message_attrs = {
            'v': '1',  # API Version
            'tid': self.tid,
            'an': self.app_name,
            'av': self.app_version,
            'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2.12.388 Version/12.14'  # dummy identifier of the browser
        }

    def send(self, message: Message):
        if self.cid is None:
            message.attrs['cid'] = str(uuid.uuid4())
        try:
            data = parse.urlencode(message.attrs).encode()

            if self.backend_url.lower().startswith('http'):
                req = request.Request(self.backend_url, data=data)
            else:
                log.info("Incorrect backend URL.")
                return

            request.urlopen(req) #nosec
        except Exception as err:
            pass  # nosec

    def build_event_message(self, event_category: str, event_action: str, event_label: str, event_value: int = 1,
                            **kwargs):
        data = self.default_message_attrs.copy()
        data.update({
            't': 'event',
            'ec': event_category,
            'ea': event_action,
            'el': event_label,
            'ev': event_value,
        })
        return Message(MessageType.EVENT, data)

    def build_session_start_message(self, category: str, **kwargs):
        data = self.default_message_attrs.copy()
        data.update({
            'sc': 'start',
            't': 'event',
            'ec': category,
            'ea': 'session',
            'el': 'start',
            'ev': 1,
        })
        return Message(MessageType.SESSION_START, data)

    def build_session_end_message(self, category: str, **kwargs):
        data = self.default_message_attrs.copy()
        data.update({
            'sc': 'end',
            't': 'event',
            'ec': category,
            'ea': 'session',
            'el': 'end',
            'ev': 1,
        })
        return Message(MessageType.SESSION_END, data)

    def build_error_message(self, category: str, error_msg: str, **kwargs):
        data = self.default_message_attrs.copy()
        data.update({
            't': 'event',
            'ec': category,
            'ea': 'error',
            'el': error_msg,
            'ev': 1,
        })
        return Message(MessageType.ERROR, data)

    def build_stack_trace_message(self, category: str, error_msg: str, **kwargs):
        data = self.default_message_attrs.copy()
        data.update({
            't': 'event',
            'ec': category,
            'ea': 'stack_trace',
            'el': error_msg,
            'ev': 1,
        })
        return Message(MessageType.STACK_TRACE, data)

    def remove_cid_file(self):
        self.cid = None
        self.default_message_attrs['cid'] = None
        remove_cid_file(self.cid_filename)

    def generate_new_cid_file(self):
        self.cid = get_or_generate_cid(self.cid_filename, lambda: str(uuid.uuid4()), is_valid_cid)
        self.default_message_attrs['cid'] = self.cid

    def cid_file_initialized(self):
        return self.cid is not None

    def set_stats(self, data: dict):
        pass


def is_valid_cid(cid: str):
    try:
        uuid.UUID(cid, version=4)
    except ValueError:
        return False
    return True

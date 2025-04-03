# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
import logging as log

from copy import copy
from urllib import request
import os

from .backend import TelemetryBackend
from ..utils.cid import get_or_generate_cid, remove_cid_file
from ..utils.params import telemetry_params
from platform import system


def _send_func(request_data):
    try:
        request.urlopen(request_data)  # nosec
    except Exception as err:
        pass  # nosec


def is_docker():
    def file_has_text(text, filename):
        try:
            with open(filename, encoding='utf-8') as lines:
                for line in lines:
                    if text in line:
                        return True
            return False
        except:
            return False

    return os.path.exists('/.dockerenv') or \
           file_has_text('docker', '/proc/self/cgroup') or \
           file_has_text('docker', '/proc/self/mountinfo')


class GA4Backend(TelemetryBackend):
    id = 'ga4'
    cid_filename = 'openvino_ga_cid'
    old_cid_filename = 'openvino_ga_uid'
    timeout = 3.0

    def __init__(self, tid: str = None, app_name: str = None, app_version: str = None):
        super(GA4Backend, self).__init__(tid, app_name, app_version)
        self.measurement_id = tid
        self.app_name = app_name
        self.app_version = app_version
        self.session_id = None
        self.cid = None
        self.backend_url = "https://www.google-analytics.com/mp/collect?measurement_id={}&api_secret={}".format(
            self.measurement_id, telemetry_params["api_key"])
        self.default_message_attrs = {
            'app_name': self.app_name,
            'app_version': self.app_version,
        }
        self.stats = {}

    def send(self, message: dict):
        if message is None:
            return
        try:
            data = json.dumps(message).encode()

            if self.backend_url.lower().startswith('http'):
                req = request.Request(self.backend_url, data=data)
            else:
                log.info("Incorrect backend URL.")
                return
            if system() == 'Windows':
                _send_func(req)
            else:
                # request.urlopen() may hang on Linux if there's no internet connection,
                # so we need to run it in a subprocess and terminate after timeout.

                # Usage of subprocesses on Windows cause unexpected behavior, when script
                # executes multiple times during subprocess initializing. For this reason
                # subprocess are not recommended on Windows.
                import multiprocessing
                process = multiprocessing.Process(target=_send_func, args=(req,))
                process.daemon = True
                process.start()

                process.join(self.timeout)
                if process.is_alive():
                    process.terminate()

        except Exception as err:
            pass  # nosec

    def build_event_message(self, event_category: str, event_action: str, event_label: str, event_value: int = 1,
                            app_name=None, app_version=None,
                            **kwargs):
        client_id = self.cid
        if client_id is None:
            client_id = "0"
        if self.session_id is None:
            self.generate_new_session_id()

        default_args = copy(self.default_message_attrs)
        default_args['docker'] = 'False'
        if app_name is not None:
            default_args['app_name'] = app_name
        if app_version is not None:
            default_args['app_version'] = app_version
        if is_docker():
            default_args['docker'] = 'True'

        payload = {
            "client_id": client_id,
            "non_personalized_ads": False,
            "events": [
                {
                    "name": event_action,
                    "params": {
                        "event_category": event_category,
                        "event_label": event_label,
                        "event_count": event_value,
                        "session_id": self.session_id,
                        **default_args,
                        **self.stats
                    }
                }
            ]
        }
        return payload

    def build_session_start_message(self, category: str, **kwargs):
        self.generate_new_session_id()
        return self.build_event_message(category, "session", "start", 1)

    def build_session_end_message(self, category: str, **kwargs):
        return self.build_event_message(category, "session", "end", 1)

    def build_error_message(self, category: str, error_msg: str, **kwargs):
        return self.build_event_message(category, "error_", error_msg, 1)

    def build_stack_trace_message(self, category: str, error_msg: str, **kwargs):
        return self.build_event_message(category, "stack_trace", error_msg, 1)

    def generate_new_cid_file(self):
        self.cid = get_or_generate_cid(self.cid_filename, lambda: str(uuid.uuid4()), is_valid_cid,
                                       self.old_cid_filename)

    def cid_file_initialized(self):
        return self.cid is not None

    def generate_new_session_id(self):
        self.session_id = str(uuid.uuid4())

    def remove_cid_file(self):
        self.cid = None
        remove_cid_file(self.cid_filename)
        remove_cid_file(self.old_cid_filename)

    def set_stats(self, data: dict):
        self.stats = data


def is_valid_cid(cid: str):
    try:
        uuid.UUID(cid, version=4)
    except ValueError:
        return False
    return True

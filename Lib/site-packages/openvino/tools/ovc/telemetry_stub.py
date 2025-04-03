# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class Telemetry(object):
    """
    Stab file for the Telemetry class which is used when Telemetry class is not available.
    """

    def __init__(self, *arg, **kwargs):
        pass

    def send_event(self, *arg, **kwargs):
        pass

    def send_error(self, *arg, **kwargs):
        pass

    def start_session(self, *arg, **kwargs):
        pass

    def end_session(self, *arg, **kwargs):
        pass

    def force_shutdown(self, *arg, **kwargs):
        pass

    def send_stack_trace(self, *arg, **kwargs):
        pass

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re


class BasicError(Exception):
    """ Base class for all exceptions in Model Conversion API

        It operates like Exception but when it is converted to str,
        it formats string as args[0].format(*args[1:]), where
        args are arguments provided when an exception instance is
        created.
    """

    def __str__(self):
        if len(self.args) <= 1:
            return Exception.__str__(self)
        return self.args[0].format(*self.args[1:])  # pylint: disable=unsubscriptable-object


class FrameworkError(BasicError):
    """ User-friendly error: raised when the error on the framework side. """
    pass


class Error(BasicError):
    """ User-friendly error: raised when the error on the user side. """
    pass


class InternalError(BasicError):
    """ Not user-friendly error: user cannot fix it and it points to the bug inside MO. """
    pass


def classify_error_type(e):
    patterns = [
        # Example: No module named 'openvino._offline_transformations.offline_transformations_api'
        r"No module named \'\S+\'",
        # Example: cannot import name 'IECore' from 'openvino.inference_engine' (unknown location)
        r"cannot import name \'\S+\'",
    ]
    error_message = str(e)
    for pattern in patterns:
        m = re.search(pattern, error_message)
        if m:
            return m.group(0)
    return "undefined"

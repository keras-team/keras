# SPDX-License-Identifier: Apache-2.0


"""
A wrapper of built-in logging with custom level support and utilities.
"""

from contextlib import contextmanager
import logging as _logging
from logging import *  # pylint: disable=wildcard-import, unused-wildcard-import
import os
import types

import tensorflow as tf
TF2 = tf.__version__.startswith("2.")

from . import constants  # pylint: disable=wrong-import-position

VERBOSE = 15

_logging.addLevelName(VERBOSE, "VERBOSE")


def _verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)  # pylint: disable=protected-access


def getLogger(name=None):  # pylint: disable=invalid-name, function-redefined
    logger = _logging.getLogger(name)
    # Inject verbose method to logger object instead logging module
    logger.verbose = types.MethodType(_verbose, logger)
    return logger


_BASIC_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_VERBOSE_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"


def basicConfig(**kwargs):  # pylint: disable=invalid-name, function-redefined
    """ Do basic configuration for the logging system. tf verbosity is updated accordingly. """
    # Choose pre-defined format if format argument is not specified
    if "format" not in kwargs:
        level = kwargs.get("level", _logging.root.level)
        kwargs["format"] = _BASIC_LOG_FORMAT if level >= INFO else _VERBOSE_LOG_FORMAT
    # config will make effect only when root.handlers is empty, so add the following statement to make sure it
    _logging.root.handlers = []
    _logging.basicConfig(**kwargs)
    set_tf_verbosity(_logging.getLogger().getEffectiveLevel())


_LOG_LEVELS = [FATAL, ERROR, WARNING, INFO, VERBOSE, DEBUG]


def get_verbosity_level(verbosity, base_level=INFO):
    """ If verbosity is specified, return corresponding level, otherwise, return default_level. """
    if verbosity is None:
        return base_level
    verbosity = min(max(0, verbosity) + _LOG_LEVELS.index(base_level), len(_LOG_LEVELS) - 1)
    return _LOG_LEVELS[verbosity]


def set_level(level):
    """ Set logging level for tf2onnx package. tf verbosity is updated accordingly. """
    _logging.getLogger(constants.TF2ONNX_PACKAGE_NAME).setLevel(level)
    set_tf_verbosity(level)


def set_tf_verbosity(level):
    """ Set TF logging verbosity."""
    # TF log is too verbose, adjust it
    if TF2:
        return

    level = ERROR if level >= INFO else level
    tf.logging.set_verbosity(level)

    # TF_CPP_MIN_LOG_LEVEL:
    #   0 = all messages are logged (default behavior)
    #   1 = INFO messages are not printed
    #   2 = INFO and WARNING messages are not printed
    #   3 = INFO, WARNING, and ERROR messages are not printed
    if level <= INFO:
        tf_cpp_min_log_level = "0"
    elif level <= WARNING:
        tf_cpp_min_log_level = "1"
    elif level <= ERROR:
        tf_cpp_min_log_level = "2"
    else:
        tf_cpp_min_log_level = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_cpp_min_log_level


@contextmanager
def set_scope_level(level, logger=None):
    """
    Set logging level to logger within context, reset level to previous value when exit context.
    TF verbosity is NOT affected.
    """
    if logger is None:
        logger = getLogger()

    current_level = logger.level
    logger.setLevel(level)

    try:
        yield logger
    finally:
        logger.setLevel(current_level)

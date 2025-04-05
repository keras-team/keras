# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging as log
import os
import re
from copy import copy

# WA for abseil bug that affects logging while importing TF starting 1.14 version
# Link to original issue: https://github.com/abseil/abseil-py/issues/99
if importlib.util.find_spec('absl') is not None:
    import absl.logging # pylint: disable=import-error

    log.root.removeHandler(absl.logging._absl_handler)

handler_num = 0


class LvlFormatter(log.Formatter):
    format_dict = {
        log.DEBUG: "[ %(asctime)s ] [ %(levelname)s ] [ %(module)s:%(lineno)d ]  %(msg)s",
        log.INFO: "[ %(levelname)s ]  %(msg)s",
        log.WARNING: "[ WARNING ]  %(msg)s",
        log.ERROR: "[ %(levelname)s ]  %(msg)s",
        log.CRITICAL: "[ %(levelname)s ]  %(msg)s",
        'framework_error': "[ FRAMEWORK ERROR ]  %(msg)s",
        'analysis_info': "[ ANALYSIS INFO ]  %(msg)s"
    }

    def __init__(self, lvl, fmt=None):
        log.Formatter.__init__(self, fmt)
        self.lvl = lvl

    def format(self, record: log.LogRecord):
        if self.lvl == 'DEBUG':
            self._style._fmt = self.format_dict[log.DEBUG]
        else:
            self._style._fmt = self.format_dict[record.levelno]
        if 'is_warning' in record.__dict__.keys():
            self._style._fmt = self.format_dict[log.WARNING]
        if 'framework_error' in record.__dict__.keys():
            self._style._fmt = self.format_dict['framework_error']
        if 'analysis_info' in record.__dict__.keys():
            self._style._fmt = self.format_dict['analysis_info']
        return log.Formatter.format(self, record)


class TagFilter(log.Filter):
    def __init__(self, regex: str):
        self.regex = regex

    def filter(self, record: log.LogRecord):
        if record.__dict__['funcName'] == 'load_grammar':  # for nx not to log into our logs
            return False
        if self.regex:
            if 'tag' in record.__dict__.keys():
                tag = record.__dict__['tag']
                return re.findall(self.regex, tag)
            else:
                return False
        return True  # if regex wasn't set print all logs


def init_logger(lvl: str, verbose: bool, python_api_used: bool):
    if verbose and python_api_used:
        # We need to not override logger in case of verbose=True to allow user set a log level
        return
    global handler_num
    log_exp = os.environ.get('MO_LOG_PATTERN')
    if not verbose:
        lvl = 'ERROR'
    fmt = LvlFormatter(lvl=lvl)
    handler = log.StreamHandler()
    handler.setFormatter(fmt)
    logger = log.getLogger()
    logger.setLevel(lvl)
    logger.addFilter(TagFilter(regex=log_exp))
    if handler_num == 0 and len(logger.handlers) == 0:
        logger.addHandler(handler)
        handler_num += 1


def get_logger_state():
    logger = log.getLogger()
    return logger.level, copy(logger.filters), copy(logger.handlers)


def restore_logger_state(state: tuple):
    level, filters, handlers = state
    logger = log.getLogger()
    logger.setLevel(level)
    logger.filters = filters
    logger.handlers = handlers

# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

import logging
import os

from .null_handler import NULLHandler

_default_log_level = os.environ.get("PYREADLINE_LOG", "DEBUG")

LOGGER = logging.getLogger("PYREADLINE")
LOGGER.setLevel(_default_log_level)
LOGGER.propagate = False
LOGGER.addHandler(NULLHandler())

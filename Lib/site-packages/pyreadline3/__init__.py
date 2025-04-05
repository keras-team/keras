# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


from platform import system
from importlib.metadata import version, PackageNotFoundError

from . import (
    clipboard,
    console,
    lineeditor,
    logger,
    modes,
    rlmain,
    unicode_helper,
)
from .rlmain import *

_S = system()

if _S.lower() != "windows":
    raise RuntimeError("pyreadline3 is for Windows only, not {}.".format(_S))

del system, _S

try:
    __version__ = version("pyreadline3")
except PackageNotFoundError:
    # package is not installed
    pass

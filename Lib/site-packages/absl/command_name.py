# Copyright 2017 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A tiny stand alone library to change the kernel process name on Linux."""

import os
import sys

# This library must be kept small and stand alone.  It is used by small things
# that require no extension modules.


def make_process_name_useful():
  """Sets the process name to something better than 'python' if possible."""
  set_kernel_process_name(os.path.basename(sys.argv[0]))


def set_kernel_process_name(name):
  """Changes the Kernel's /proc/self/status process name on Linux.

  The kernel name is NOT what will be shown by the ps or top command.
  It is a 15 character string stored in the kernel's process table that
  is included in the kernel log when a process is OOM killed.
  The first 15 bytes of name are used.  Non-ASCII unicode is replaced with '?'.

  Does nothing if /proc/self/comm cannot be written or prctl() fails.

  Args:
    name: bytes|unicode, the Linux kernel's command name to set.
  """
  if not isinstance(name, bytes):
    name = name.encode('ascii', 'replace')
  try:
    # This is preferred to using ctypes to try and call prctl() when possible.
    with open('/proc/self/comm', 'wb') as proc_comm:
      proc_comm.write(name[:15])
  except OSError:
    try:
      import ctypes  # pylint: disable=g-import-not-at-top
    except ImportError:
      return  # No ctypes.
    try:
      libc = ctypes.CDLL('libc.so.6')
    except OSError:
      return  # No libc.so.6.
    pr_set_name = ctypes.c_ulong(15)  # linux/prctl.h PR_SET_NAME value.
    zero = ctypes.c_ulong(0)
    try:
      libc.prctl(pr_set_name, name, zero, zero, zero)
      # Ignore the prctl return value.  Nothing we can do if it errored.
    except AttributeError:
      return  # No prctl.

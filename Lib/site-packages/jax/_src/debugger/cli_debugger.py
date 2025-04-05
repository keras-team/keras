# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import cmd
import pprint
import sys
import traceback
from typing import Any, IO

from jax._src.debugger import core as debugger_core

DebuggerFrame = debugger_core.DebuggerFrame

class CliDebugger(cmd.Cmd):
  """A text-based debugger."""
  prompt = '(jdb) '

  def __init__(self, frames: list[DebuggerFrame], thread_id,
      stdin: IO[str] | None = None, stdout: IO[str] | None = None,
      completekey: str = "tab"):
    super().__init__(stdin=stdin, stdout=stdout, completekey=completekey)
    self.use_rawinput = stdin is None
    self.frames = frames
    self.frame_index = 0
    self.thread_id = thread_id
    self.intro = 'Entering jdb:'

  def current_frame(self):
    return self.frames[self.frame_index]

  def evaluate(self, expr):
    env = {}
    curr_frame = self.frames[self.frame_index]
    env.update(curr_frame.globals)
    env.update(curr_frame.locals)
    return eval(expr, {}, env)

  def default(self, arg):
    """Evaluates an expression."""
    try:
      print(repr(self.evaluate(arg)), file=self.stdout)
    except:
      self._error_message()

  def print_backtrace(self):
    backtrace = []
    backtrace.append('Traceback:')
    for frame in self.frames[::-1]:
      backtrace.append(f'  File "{frame.filename}", line {frame.lineno}')
      if frame.offset is None:
        backtrace.append('    <no source>')
      else:
        line = frame.source[frame.offset]
        backtrace.append(f'    {line.strip()}')
    print("\n".join(backtrace), file=self.stdout)

  def print_context(self, num_lines=2):
    curr_frame = self.frames[self.frame_index]
    context = []
    context.append(f'> {curr_frame.filename}({curr_frame.lineno})')
    for i, line in enumerate(curr_frame.source):
      assert curr_frame.offset is not None
      if (curr_frame.offset - 1 - num_lines <= i <=
          curr_frame.offset + num_lines):
        if i == curr_frame.offset:
          context.append(f'->  {line}')
        else:
          context.append(f'    {line}')
    print("\n".join(context), file=self.stdout)

  def _error_message(self):
    exc_info = sys.exc_info()[:2]
    msg = traceback.format_exception_only(*exc_info)[-1].strip()
    print('***', msg, file=self.stdout)

  def do_p(self, arg):
    """p expression
    Evaluates and prints the value of an expression
    """
    try:
      print(repr(self.evaluate(arg)), file=self.stdout)
    except:
      self._error_message()

  def do_pp(self, arg):
    """pp expression
    Evaluates and pretty-prints the value of an expression
    """
    try:
      print(pprint.pformat(self.evaluate(arg)), file=self.stdout)
    except:
      self._error_message()

  def do_up(self, _):
    """u(p)
    Move down a stack frame.
    """
    if self.frame_index == len(self.frames) - 1:
      print('At topmost frame.', file=self.stdout)
    else:
      self.frame_index += 1
    self.print_context()
  do_u = do_up

  def do_down(self, _):
    """d(own)
    Move down a stack frame.
    """
    if self.frame_index == 0:
      print('At bottommost frame.', file=self.stdout)
    else:
      self.frame_index -= 1
    self.print_context()
  do_d = do_down

  def do_list(self, _):
    """l(ist)
    List source code for the current file.
    """
    self.print_context(num_lines=5)
  do_l = do_list

  def do_continue(self, _):
    """c(ont(inue))
    Continue the program's execution.
    """
    return True
  do_c = do_cont = do_continue

  def do_quit(self, _):
    """q(uit)\n(exit)
    Quit the debugger. The program is given an exit command.
    """
    sys.exit(0)
  do_q = do_EOF = do_exit = do_quit

  def do_where(self, _):
    """w(here)
    Prints a stack trace with the most recent frame on the bottom.
    'bt' is an alias for this command.
    """
    self.print_backtrace()
  do_w = do_bt = do_where

  def run(self):
    while True:
      try:
        self.cmdloop()
        break
      except KeyboardInterrupt:
        print('--KeyboardInterrupt--', file=sys.stdout)

def run_debugger(frames: list[DebuggerFrame], thread_id: int | None,
                 **kwargs: Any):
  CliDebugger(frames, thread_id, **kwargs).run()
debugger_core.register_debugger("cli", run_debugger, -1)

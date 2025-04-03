# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorBoard main module.

This module ties together `tensorboard.program` and
`tensorboard.default_plugins` to provide standard TensorBoard. It's
meant to be tiny and act as little other than a config file. Those
wishing to customize the set of plugins or static assets that
TensorBoard uses can swap out this file with their own.
"""

import sys

from absl import app
from tensorboard import default
from tensorboard import main_lib
from tensorboard import program
from tensorboard.plugins import base_plugin
from tensorboard.util import tb_logging

logger = tb_logging.get_logger()


def run_main():
    """Initializes flags and calls main()."""
    main_lib.global_init()
    tensorboard = program.TensorBoard(plugins=default.get_plugins())
    try:
        app.run(tensorboard.main, flags_parser=tensorboard.configure)
    except base_plugin.FlagsError as e:
        print("Error: %s" % e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_main()

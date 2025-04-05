# Copyright 2024 The etils Authors.
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

"""Colab public API."""

# pylint: disable=g-importing-member

from etils.ecolab.array_as_img import auto_plot_array
from etils.ecolab.auto_display_utils import auto_display
from etils.ecolab.auto_display_utils import disp
from etils.ecolab.colab_utils import collapse
from etils.ecolab.colab_utils import get_permalink
from etils.ecolab.colab_utils import interruptible
from etils.ecolab.colab_utils import json
from etils.ecolab.highlight_util import highlight_html
from etils.ecolab.inplace_reload import ReloadMode
from etils.ecolab.inspects.auto_utils import auto_inspect
from etils.ecolab.inspects.core import inspect
from etils.ecolab.patch_utils import patch_graphviz
from etils.ecolab.patch_utils import set_verbose
from etils.ecolab.pyjs_com import js_import as pyjs_import
from etils.ecolab.pyjs_com import register_js_fn
from etils.epy.adhoc_utils.module_utils import clear_cached_modules

# Activate auto-display by default
auto_display()

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

"""XManager utils."""

# pylint: disable=g-importing-member

from etils.exm.dummy import current_experiment
from etils.exm.dummy import current_work_unit
from etils.exm.dummy import is_running_under_xmanager
from etils.exm.dummy import add_experiment_artifact
from etils.exm.dummy import add_work_unit_artifact
from etils.exm.dummy import curr_job_name
from etils.exm.dummy import url_to_python_only_logs
from etils.exm.dummy import set_citc_source

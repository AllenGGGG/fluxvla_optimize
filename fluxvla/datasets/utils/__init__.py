# Copyright 2026 Limx Dynamics
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

import importlib
import warnings

# Required for training (save_dataset_statistics et al.); import eagerly. Its
# heavy RLDS deps (dlimp/tensorflow) are imported best-effort inside the module.
from .data_utils import *  # noqa: F401, F403

# Remaining utils pull in heavy RLDS/sim deps at import time and are not used by
# the parquet FSDP training path. Import best-effort so training-only envs are
# not blocked; registration happens as a side effect of import.
for _mod_name in ('data_transforms', 'goal_relabeling',
                  'sarm_utils', 'task_augmentation'):
    try:
        importlib.import_module(f'.{_mod_name}', __name__)
    except Exception as _exc:  # noqa: BLE001
        warnings.warn(
            f'Optional dataset util {_mod_name} not available: {_exc}',
            stacklevel=2)

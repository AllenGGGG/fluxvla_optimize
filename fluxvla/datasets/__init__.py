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

# Core datasets required for parquet training; import eagerly.
from .dataset_wrapper import *  # noqa: F401, F403
from .packed_parquet_dataset_v3 import *  # noqa: F401, F403
from .parquet_dataset_v3 import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403

# Optional datasets that pull in heavy deps (tensorflow for RLDS, sim/annotation
# stacks for SARM/online). Import best-effort so a training-only environment is
# not blocked; registration happens as a side effect of import.
for _mod_name in ('online_vsta_dataset', 'parquet_dataset', 'rlds_dataset',
                  'sarm_dataset'):
    try:
        importlib.import_module(f'.{_mod_name}', __name__)
    except Exception as _exc:  # noqa: BLE001
        warnings.warn(
            f'Optional dataset {_mod_name} not available: {_exc}', stacklevel=2)

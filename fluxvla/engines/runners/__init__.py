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

# Training runners are required; import them eagerly so failures surface.
from .base_train_runner import BaseTrainRunner  # noqa: F401, F403
from .ddp_train_runner import DDPTrainRunner  # noqa: F401, F403
from .fsdp_train_runner import FSDPTrainRunner  # noqa: F401, F403

# Simulation / real-robot eval and inference runners pull in optional heavy
# dependencies (libero, robosuite, mujoco, robot SDKs) that are not needed for
# dataset training. Import them best-effort so a training-only environment is
# not blocked by missing optional packages.
_OPTIONAL_RUNNERS = {
    'aloha_inference_runner': ['AlohaInferenceRunner'],
    'aloha_rtc_inference_runner': ['AlohaRTCInferenceRunner'],
    'fluxbisim_aloha_inference_runner': ['AlohaInferenceRunnerSim'],
    'fluxbisim_base_inference_runner': ['BaseInferenceRunnerSim'],
    'libero_eval_runner': ['LiberoEvalRunner'],
    'libero_inference_runner': ['LiberoInferenceRunner'],
    'tron2_inference_runner': ['Tron2InferenceRunner'],
    'tron2_rtc_inference_runner': ['Tron2RTCInferenceRunner'],
    'ur_inference_runner': ['URInferenceRunner'],
}
for _mod_name, _names in _OPTIONAL_RUNNERS.items():
    try:
        _mod = importlib.import_module(f'.{_mod_name}', __name__)
    except Exception as _exc:  # noqa: BLE001
        warnings.warn(
            f'Optional runner {_mod_name} not available: {_exc}', stacklevel=2)
        continue
    for _name in _names:
        globals()[_name] = getattr(_mod, _name)

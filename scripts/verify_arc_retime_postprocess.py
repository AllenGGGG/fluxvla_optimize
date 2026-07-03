#!/usr/bin/env python3
"""Regression checks for LIBERO arc/retime action postprocess.

This script intentionally avoids importing the full FluxVLA package.  The
runner module has heavy top-level robotics/model imports, while the tested
postprocess helpers are pure NumPy functions.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


class _RegistryStub:

    def register_module(self):
        return lambda cls: cls


def _install_import_stubs():
    libero = types.ModuleType('libero')
    libero.libero = types.ModuleType('libero.libero')
    libero.libero.benchmark = types.SimpleNamespace()
    sys.modules['libero'] = libero
    sys.modules['libero.libero'] = libero.libero

    safetensors = types.ModuleType('safetensors')
    safetensors.torch = types.ModuleType('safetensors.torch')
    safetensors.torch.load_file = lambda *args, **kwargs: None
    sys.modules['safetensors'] = safetensors
    sys.modules['safetensors.torch'] = safetensors.torch

    for name in [
            'fluxvla',
            'fluxvla.engines',
            'fluxvla.engines.runners',
            'fluxvla.engines.utils',
    ]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    utils = sys.modules['fluxvla.engines.utils']
    utils.initialize_overwatch = lambda *args, **kwargs: types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        rank=lambda: 0,
        world_size=lambda: 1,
    )

    name_map = types.ModuleType('fluxvla.engines.utils.name_map')
    name_map.str_to_dtype = lambda value: value
    sys.modules['fluxvla.engines.utils.name_map'] = name_map

    torch_utils = types.ModuleType('fluxvla.engines.utils.torch_utils')
    torch_utils.set_seed_everywhere = lambda *args, **kwargs: None
    sys.modules['fluxvla.engines.utils.torch_utils'] = torch_utils

    root = types.ModuleType('fluxvla.engines.utils.root')
    root.RUNNERS = _RegistryStub()
    sys.modules['fluxvla.engines.utils.root'] = root


def _load_runner_class():
    _install_import_stubs()
    repo_root = Path(__file__).resolve().parents[1]
    runner_path = repo_root / 'fluxvla/engines/runners/libero_eval_runner.py'
    spec = importlib.util.spec_from_file_location(
        'fluxvla.engines.runners.libero_eval_runner_under_test',
        runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LiberoEvalRunner


def main():
    runner = _load_runner_class()
    cont_idx = [0, 1, 2, 3, 4, 5]
    gripper_idx = [6]

    for seed in range(20):
        rng = np.random.default_rng(seed)
        actions = rng.normal(size=(10, 7)).astype(np.float32)
        if seed % 3 == 0:
            actions[4:, 6] += 1.0

        strict_guard = dict(
            fallback_until_feasible=True,
            allow_zero_motion_passthrough=False,
            max_delta_ratio=0.001,
            max_accel_ratio=0.001,
            cont_idx=cont_idx,
            gripper_idx=gripper_idx,
        )
        for once_fn in (runner._arc_retime_once, runner._rdp_delta_once):
            out = runner._guarded_delta_retime(
                actions, cont_idx, gripper_idx, 6, strict_guard, once_fn)
            assert out.shape == actions.shape, (seed, once_fn.__name__,
                                                out.shape)
            assert np.array_equal(out, actions), (seed, once_fn.__name__)

        adaptive_identity_cfg = dict(
            min_output_steps=10,
            max_output_steps=10,
            fallback_until_feasible=True,
            cont_idx=cont_idx,
            gripper_idx=gripper_idx,
            path_fidelity_guard=True,
            first_step_guard=True,
            protect_gripper_change=True,
            protected_min_output_steps=10,
        )
        out = runner._arc_adaptive_delta(
            actions, cont_idx, gripper_idx, 6, adaptive_identity_cfg)
        assert out.shape == actions.shape, (seed, out.shape)
        assert np.array_equal(out, actions), seed

        adaptive_compress_cfg = dict(
            min_output_steps=6,
            max_output_steps=6,
            fallback_until_feasible=False,
            cont_idx=cont_idx,
            gripper_idx=gripper_idx,
            path_fidelity_guard=False,
            first_step_guard=False,
        )
        out = runner._arc_adaptive_delta(
            actions, cont_idx, gripper_idx, 6, adaptive_compress_cfg)
        assert out.shape == (6, 7), (seed, out.shape)

    print('arc/retime postprocess regression ok: 20 seeds')


if __name__ == '__main__':
    main()

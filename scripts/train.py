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

import argparse
import json
import os
import random
import sys
import warnings

# Use the checkout that owns this script even when another FluxVLA checkout is
# installed editable in the selected conda environment.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings('ignore', category=UnsupportedFieldAttributeWarning)

import draccus
import numpy as np
import torch
import yaml
from mmengine import Config, DictAction

from fluxvla.datasets.utils import (save_dataset_statistics,
                                    save_grouped_dataset_statistics)
from fluxvla.engines import (build_dataset_from_cfg, build_runner_from_cfg,
                             initialize_overwatch)
from fluxvla.engines.utils.torch_utils import set_global_seed

overwatch = initialize_overwatch(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model with the given configuration.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file.',
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='The directory to save logs and checkpoints.',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='The directory to save metric logs. Defaults to --work-dir.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=  # noqa: E251
        'override some settings in the used config, the key-value pair in xxx=yyy format'  # noqa: E501
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to the checkpoint file to resume training from.',
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def _save_resolved_config(cfg, work_dir):
    config_yaml_path = os.path.join(work_dir, 'config.yaml')
    config_json_path = os.path.join(work_dir, 'config.json')

    with open(config_yaml_path, 'w') as f_yaml:
        draccus.dump(cfg.to_dict(), f_yaml)

    with open(config_yaml_path, 'r') as f_yaml, open(config_json_path,
                                                     'w') as f_json:
        yaml_cfg = yaml.safe_load(f_yaml)
        json.dump(yaml_cfg, f_json, indent=2)


def _get_nested_value(obj, path):
    for key in path:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
    return obj


def _resolve_train_seed(cfg):
    """Resolve an explicit training seed from existing config fields."""
    # Dataset-level seeds control dataset sampling order and may already exist
    # in older configs; do not promote them to global training seeds.
    for path in (
        ('runner', 'seed'),
        ('train_dataloader', 'seed'),
        ('seed', ),
    ):
        seed = _get_nested_value(cfg, path)
        if seed is not None:
            return int(seed)
    return None


def _set_rank_training_seed(seed):
    rank_seed = int(seed) + overwatch.rank()
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    return rank_seed


def train(args, cfg):
    """Train the model with the given configuration.

    Args:
        cfg (Config): The configuration object containing training settings.
    """
    seed = _resolve_train_seed(cfg)
    if seed is not None:
        set_global_seed(seed)
        if overwatch.is_rank_zero():
            overwatch.info(f'Training seed set to {seed}.')

    log_dir = args.log_dir or args.work_dir
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    dataset = build_dataset_from_cfg(cfg.train_dataloader.dataset)
    if overwatch.is_rank_zero() and hasattr(dataset, 'dataset_statistics'):
        save_dataset_statistics(dataset.dataset_statistics, args.work_dir)
    elif overwatch.is_rank_zero() and hasattr(dataset,
                                              'grouped_dataset_statistics'):
        # Handle grouped dataset statistics
        save_grouped_dataset_statistics(dataset.grouped_dataset_statistics,
                                        args.work_dir)
    if overwatch.is_rank_zero():
        _save_resolved_config(cfg, args.work_dir)
    if overwatch.is_rank_zero() and hasattr(
            dataset, 'batch_transform') and hasattr(
                dataset.batch_transform, 'base_tokenizer'):  # noqa: E501
        tokenizer = dataset.batch_transform.base_tokenizer
        tokenizer.save_pretrained(args.work_dir)
        overwatch.info(f'Saved tokenizer to {args.work_dir}')
    if hasattr(cfg.runner, 'metric'):
        cfg.runner.metric.run_dir = log_dir
    cfg.runner.checkpoint_run_dir = args.work_dir
    cfg.runner.cfg = cfg
    cfg.runner.args = args
    if args.resume_from is not None:
        cfg.runner.resume_from = args.resume_from
    runner = build_runner_from_cfg(cfg.runner)  # noqa: F841
    runner.run_setup(n_train_examples=len(dataset))
    if seed is not None:
        rank_seed = _set_rank_training_seed(seed)
        if overwatch.is_rank_zero():
            overwatch.info('Training RNG reset after model setup; '
                           f'rank-local base seed is {rank_seed}.')
    runner.run(dataset)


if __name__ == '__main__':
    args, _ = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    train(args, cfg)

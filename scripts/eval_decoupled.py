#!/usr/bin/env python3
"""Run LIBERO eval with decoupled full/decoder-only alternating inference.

This is a thin wrapper around the standard eval runner. It preserves the
normal dataset preprocessing, action denormalization, distributed execution,
and logging path, while enabling the decoupled alternating mode.
"""

import argparse

from mmengine import Config, DictAction

from fluxvla.engines import build_runner_from_cfg, initialize_overwatch

overwatch = initialize_overwatch(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate with decoupled full/decoder-only inference.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file.',
    )
    parser.add_argument(
        '--ckpt-path',
        type=str,
        default=None,
        help='Path to the checkpoint file.')
    parser.add_argument(
        '--exec-chunk-size',
        type=int,
        default=None,
        help='Number of actions to execute per full/decoder-only call.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, using key=value format.')
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == '__main__':
    args, _ = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.inference_model.get('type') != 'PI05FlowMatchingDecoupledInference':
        cfg.inference_model.type = 'PI05FlowMatchingDecoupledInference'
    if 'exec_chunk_size' not in cfg.inference_model:
        cfg.inference_model.exec_chunk_size = (
            args.exec_chunk_size if args.exec_chunk_size is not None else 5)

    cfg.eval.decoupled_alternating = True
    cfg.eval.decoupled_exec_chunk_size = (
        args.exec_chunk_size
        if args.exec_chunk_size is not None else
        cfg.inference_model.get('exec_chunk_size', 5))
    cfg.eval.cfg = cfg
    cfg.eval.ckpt_path = args.ckpt_path
    if hasattr(cfg.eval,
               'processor') and not hasattr(cfg.eval.processor, 'model_path'):
        cfg.eval.processor.model_path = args.ckpt_path

    eval_runner = build_runner_from_cfg(cfg.eval)
    eval_runner.run_setup()
    eval_runner.run()

#!/usr/bin/env python3
"""Print real LIBERO prompt length and PI0.5 Triton graph lengths."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from mmengine import Config
from safetensors.torch import load_file

DEFAULT_WITH_L_CONFIG = 'configs/pi05/pi05_paligemma_libero_10_full_inference.py'
DEFAULT_PIXSHUFFLE_CONFIG = (
    'configs/pi05/pi05_libero10_task0_with_l_pixshuffle_inference.py')
DEFAULT_WITH_L_CKPT = (
    'work_dirs/pi05_libero10_task0_with_l/checkpoints/'
    'step-008460-epoch-60-loss=0.0910.safetensors')
DEFAULT_PIXSHUFFLE_CKPT = (
    'work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/'
    'step-008460-epoch-60-loss=0.0918.safetensors')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run with_l and pixshuffle accelerated PI0.5 for a real LIBERO '
            'task prompt, then print language length, visual token length, '
            'and Triton/CUDA-graph compute lengths.'))
    parser.add_argument('--with-l-config', default=DEFAULT_WITH_L_CONFIG)
    parser.add_argument('--pixshuffle-config', default=DEFAULT_PIXSHUFFLE_CONFIG)
    parser.add_argument('--with-l-ckpt', default=DEFAULT_WITH_L_CKPT)
    parser.add_argument('--pixshuffle-ckpt', default=DEFAULT_PIXSHUFFLE_CKPT)
    parser.add_argument(
        '--base-weights',
        default='./checkpoints/pi05_base/model.safetensors',
        help='Base weights used when constructing inference_model before ckpt load.')
    parser.add_argument('--task-suite-name', default='libero_10')
    parser.add_argument('--task-id', type=int, default=4)
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=('bf16', 'fp16', 'fp32'), default='bf16')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--output-jsonl',
        default=None,
        help='Optional path for one JSON record per variant/run.')
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == 'bf16':
        return torch.bfloat16
    if name == 'fp16':
        return torch.float16
    if name == 'fp32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')


def variant_items(args: argparse.Namespace) -> List[Dict[str, str]]:
    return [
        {
            'variant': 'with_l',
            'config': args.with_l_config,
            'ckpt': args.with_l_ckpt,
        },
        {
            'variant': 'pixshuffle',
            'config': args.pixshuffle_config,
            'ckpt': args.pixshuffle_ckpt,
        },
    ]


def find_prompt_transform_cfg(cfg: Config) -> Dict:
    for transform in cfg.eval.dataset.transforms:
        if transform.get('type') in ('LiberoPromptFromInputs',
                                     'NoLanguagePrompt'):
            return dict(transform)
    raise ValueError('No Libero prompt transform found in cfg.eval.dataset.')


def prompt_text_from_cfg(transform_cfg: Dict, task_description: str) -> str:
    use_conversation = transform_cfg.get('use_conversation', True)
    prompt_suffix = transform_cfg.get('prompt_suffix', '')
    add_new_line = transform_cfg.get('add_new_line', False)
    if use_conversation:
        prompt = ('In: What action should the robot take to ' +
                  str(task_description).lower() + '?\nOut:' + prompt_suffix)
    else:
        prompt = task_description
    if add_new_line:
        prompt += '\n'
    return prompt


def get_task_description(task_suite_name: str, task_id: int) -> str:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    return task.language


def tokenize_real_prompt(config_path: str, task_suite_name: str,
                         task_id: int) -> Dict:
    import fluxvla  # noqa: F401
    from fluxvla.engines import build_transform_from_cfg

    cfg = Config.fromfile(config_path)
    transform_cfg = find_prompt_transform_cfg(cfg)
    task_description = get_task_description(task_suite_name, task_id)
    prompt_text = prompt_text_from_cfg(transform_cfg, task_description)
    prompt_transform = build_transform_from_cfg(transform_cfg)
    data = prompt_transform({'task_description': task_description})
    lang_tokens = torch.tensor(data['lang_tokens'], dtype=torch.long).unsqueeze(0)
    lang_masks = torch.tensor(data['lang_masks'], dtype=torch.bool).unsqueeze(0)
    return {
        'task_description': task_description,
        'prompt_text': prompt_text,
        'lang_tokens_cpu': lang_tokens,
        'lang_masks_cpu': lang_masks,
        'lang_tokens_shape': list(lang_tokens.shape),
        'lang_alloc_len': int(lang_tokens.shape[-1]),
        'lang_actual_len': int(lang_masks[0].sum().item()),
        'prompt_transform': transform_cfg,
    }


def load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    if ckpt_path.endswith('.safetensors'):
        return load_file(ckpt_path, device='cpu')
    checkpoint = torch.load(ckpt_path, map_location='cpu', mmap=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def build_model(config_path: str, ckpt_path: str, args: argparse.Namespace):
    import fluxvla  # noqa: F401
    from fluxvla.engines import build_vla_from_cfg

    cfg = Config.fromfile(config_path)
    cfg.inference_model.use_language = True
    cfg.inference_model.pretrained_name_or_path = args.base_weights
    model = build_vla_from_cfg(cfg.inference_model).eval()
    state_dict = load_state_dict(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=torch.device(args.device), dtype=dtype_from_name(args.dtype))
    model.eval()
    return model


def make_batch(prompt_info: Dict, args: argparse.Namespace,
               model) -> Dict[str, torch.Tensor]:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    num_views = int(getattr(model, 'num_views', 2))
    image_size = int(model.vision_backbone.vision.vision_model.config.image_size)
    action_steps = int(model.n_action_steps)
    action_dim = int(model.max_action_dim)
    state_dim = action_dim
    return {
        'images':
        torch.zeros(
            1,
            num_views * 3,
            image_size,
            image_size,
            dtype=dtype,
            device=device),
        'img_masks':
        torch.ones(1, num_views, dtype=torch.bool, device=device),
        'lang_tokens':
        prompt_info['lang_tokens_cpu'].to(device),
        'lang_masks':
        prompt_info['lang_masks_cpu'].to(device),
        'states':
        torch.zeros(1, state_dim, dtype=dtype, device=device),
        'noise':
        torch.zeros(1, action_steps, action_dim, dtype=dtype, device=device),
    }


def shape_of(value) -> List[int]:
    return list(value.shape)


def collect_lengths(model, prompt_info: Dict, variant: str, run_idx: int,
                    elapsed_ms: float) -> Dict:
    bufs = getattr(model, '_triton_bufs', {})
    actual_prompt_len = prompt_info['lang_actual_len']
    visual_tokens_per_view = int(getattr(model, '_visual_tokens_per_view'))
    raw_tokens_per_view = int(getattr(model, '_vit_num_patches'))
    num_views = int(getattr(model, 'num_views'))
    graph_encoder_seq_len = int(getattr(model, '_encoder_seq_len'))
    valid_encoder_len = int(bufs['valid_encoder_len'].item())
    decoder_seq_len = int(getattr(model, '_decoder_seq_len'))
    record = {
        'variant': variant,
        'run': run_idx,
        'elapsed_ms': elapsed_ms,
        'task_description': prompt_info['task_description'],
        'prompt_text': prompt_info['prompt_text'],
        'lang_tokens_shape': prompt_info['lang_tokens_shape'],
        'lang_alloc_len': prompt_info['lang_alloc_len'],
        'lang_actual_len_from_mask': actual_prompt_len,
        'triton_max_prompt_len': int(getattr(model, '_max_prompt_len')),
        'num_views': num_views,
        'vision_raw_tokens_per_view': raw_tokens_per_view,
        'vision_raw_tokens_total': num_views * raw_tokens_per_view,
        'vision_tokens_per_view_used_by_llm': visual_tokens_per_view,
        'vision_tokens_total_used_by_llm': num_views * visual_tokens_per_view,
        'valid_encoder_len_used_for_mask': valid_encoder_len,
        'graph_encoder_seq_len_used_by_triton_ops': graph_encoder_seq_len,
        'decoder_seq_len': decoder_seq_len,
        'decoder_total_keys_in_graph': graph_encoder_seq_len + decoder_seq_len,
        'encoder_x_shape': shape_of(bufs['encoder_x']),
        'encoder_q_shape': shape_of(bufs['encoder_Q']),
        'encoder_logits_buf_shape': shape_of(bufs['encoder_logits_buf']),
        'decoder_logits_buf_shape': shape_of(bufs['decoder_logits_buf']),
        'encoder_kv_shape': shape_of(bufs['encoder_K']),
        'vision_x_raw_shape': shape_of(bufs['vision_x']),
    }
    if 'vision_projector_x' in bufs:
        record['vision_projector_x_shape'] = shape_of(
            bufs['vision_projector_x'])
    return record


def print_record(record: Dict) -> None:
    print(
        '[LengthDebug] '
        f"variant={record['variant']} "
        f"run={record['run']} "
        f"elapsed_ms={record['elapsed_ms']:.3f} "
        f"lang_actual={record['lang_actual_len_from_mask']} "
        f"lang_alloc={record['lang_alloc_len']} "
        f"triton_max_prompt={record['triton_max_prompt_len']} "
        f"vision_raw_total={record['vision_raw_tokens_total']} "
        f"vision_used_total={record['vision_tokens_total_used_by_llm']} "
        f"valid_encoder_len={record['valid_encoder_len_used_for_mask']} "
        f"graph_encoder_seq_len={record['graph_encoder_seq_len_used_by_triton_ops']} "
        f"decoder_total_keys_graph={record['decoder_total_keys_in_graph']}",
        flush=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for PI0.5 Triton length debug.')

    prompt_info = tokenize_real_prompt(args.with_l_config,
                                       args.task_suite_name, args.task_id)
    print('[PromptDebug] task_description=' +
          json.dumps(prompt_info['task_description'], ensure_ascii=False))
    print('[PromptDebug] prompt_text=' +
          json.dumps(prompt_info['prompt_text'], ensure_ascii=False))
    print('[PromptDebug] lang_tokens_shape={} lang_actual_len={}'.format(
        prompt_info['lang_tokens_shape'], prompt_info['lang_actual_len']))
    print('[PromptDebug] prompt_transform=' +
          json.dumps(prompt_info['prompt_transform'], ensure_ascii=False))

    output_file = None
    if args.output_jsonl:
        output_file = Path(args.output_jsonl)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for item in variant_items(args):
        print(f"[Build] variant={item['variant']} config={item['config']}")
        model = build_model(item['config'], item['ckpt'], args)
        batch = make_batch(prompt_info, args, model)
        records = []
        for run_idx in range(1, args.num_runs + 1):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model.predict_action(**batch)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            record = collect_lengths(model, prompt_info, item['variant'],
                                     run_idx, elapsed_ms)
            print_record(record)
            records.append(record)
            all_records.append(record)
        del model
        torch.cuda.empty_cache()

        first = records[0]
        print(
            '[LengthSummary] '
            f"variant={item['variant']} "
            f"actual_prompt_len={first['lang_actual_len_from_mask']} "
            f"uses_valid_encoder_len={first['valid_encoder_len_used_for_mask']} "
            f"uses_graph_encoder_seq_len={first['graph_encoder_seq_len_used_by_triton_ops']} "
            f"visual_tokens_used={first['vision_tokens_total_used_by_llm']} "
            f"raw_vision_tokens={first['vision_raw_tokens_total']}",
            flush=True)

    if output_file is not None:
        with output_file.open('w') as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'[Done] wrote {output_file}', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Profile PI0.5 inference to identify bottlenecks."""

import argparse
import time
import torch
from mmengine import Config
from fluxvla.engines import build_vla


def profile_inference(config_path, checkpoint_path, num_iters=100, warmup=10):
    """Profile inference and report timing breakdown."""
    cfg = Config.fromfile(config_path)

    # Build model
    model = build_vla(cfg.inference_model)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.cuda().eval()

    # Prepare dummy inputs
    num_views = getattr(model, 'num_views', 2)
    batch = {
        'images': torch.randn(1, num_views * 3, 224, 224, device='cuda'),
        'lang_tokens': torch.randint(0, 1000, (1, 32), device='cuda'),
        'states': torch.randn(1, 7, device='cuda'),
        'lang_masks': torch.ones(1, 32, device='cuda', dtype=torch.bool),
    }

    # Warmup
    print(f"Warming up for {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.predict_action(**batch)

    torch.cuda.synchronize()

    # Profile
    print(f"\nProfiling {num_iters} iterations...")
    times = []

    with torch.no_grad():
        for i in range(num_iters):
            start = time.perf_counter()
            _ = model.predict_action(**batch)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{num_iters}: {times[-1]:.2f} ms")

    # Report
    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n{'='*60}")
    print(f"Results over {num_iters} iterations:")
    print(f"  Mean: {mean_time:.2f} ms")
    print(f"  Min:  {min_time:.2f} ms")
    print(f"  Max:  {max_time:.2f} ms")
    print(f"  FPS:  {1000/mean_time:.1f}")
    print(f"{'='*60}")

    # Try to get breakdown if model supports it
    if hasattr(model, '_triton_ready') and model._triton_ready:
        print("\nModel is using Triton accelerated path.")
        print("This combines vision_encoder + transformer_encoder + decoder into one CUDA Graph.")
        print("\nTo get detailed breakdown, you would need to:")
        print("  1. Disable CUDA Graph temporarily")
        print("  2. Add timing hooks in vision_encoder/transformer_encoder/transformer_decoder")
    else:
        print("\nModel is using eager mode (not Triton accelerated).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    profile_inference(args.config, args.checkpoint, args.num_iters, args.warmup)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Lightweight checks for the realtime-named TempoVLA inference entry point.

This does not claim raw RealTimeVLA kernels are production-ready.  It verifies
that the realtime config is safe for existing FluxVLA evaluation: it uses the
stable speed-modulated inference contract and keeps speed out of the prompt.
"""

from pathlib import Path

from mmengine import Config

REPO_ROOT = Path(__file__).resolve().parents[1]


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    cfg_path = (
        REPO_ROOT / 'configs' / 'pi05' /
        'pi05_libero10_task0_tempovla_realtime_inference.py')
    wrapper_path = (
        REPO_ROOT / 'fluxvla' / 'models' / 'vlas' /
        'pi05_realtime_speed_modulated_inference.py')
    base_path = (
        REPO_ROOT / 'fluxvla' / 'models' / 'vlas' /
        'pi05_realtime_base.py')

    cfg = Config.fromfile(str(cfg_path))
    assert_true(hasattr(cfg, 'inference_model'),
                'config must define inference_model')
    assert_true(
        cfg.inference_model.type == 'PI05RealTimeSpeedModulatedInference',
        'config must use PI05RealTimeSpeedModulatedInference')
    assert_true(
        cfg.inference_model.get('enable_realtime_kernels', False) is False,
        'raw realtime kernels must be opt-in only')

    prompt_transform = cfg.eval.dataset.transforms[2]
    assert_true(prompt_transform.get('type') == 'LiberoPromptFromInputs',
                'unexpected prompt transform')
    assert_true('speed' not in prompt_transform,
                'speed must not be injected into the prompt')
    assert_true('speed_prompt_template' not in prompt_transform,
                'speed prompt template must not be present')

    wrapper_src = wrapper_path.read_text()
    assert_true('PI05FlowMatchingSpeedModulatedInference' in wrapper_src,
                'wrapper must inherit the stable speed-modulated inference')
    assert_true('enable_realtime_kernels' in wrapper_src,
                'wrapper must expose explicit realtime opt-in flag')

    base_src = base_path.read_text()
    assert_true('from .pi0_infer import' in base_src,
                'pi05_realtime_base.py must use package-relative pi0_infer')

    print('realtime TempoVLA safety checks passed')


if __name__ == '__main__':
    main()


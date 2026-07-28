from collections import OrderedDict
from types import SimpleNamespace
import unittest

import torch

from fluxvla.engines.runners.base_train_runner import BaseTrainRunner


class _RunnerForTest(BaseTrainRunner):

    def save_checkpoint(self, *args, **kwargs):
        pass

    def clip_grad_norm(self):
        return None

    def _load_model_state(self, *args, **kwargs):
        pass

    def _load_optimizer_state(self, *args, **kwargs):
        pass


def _runner():
    runner = object.__new__(_RunnerForTest)
    runner.metric = SimpleNamespace(global_step=9)
    runner.vla = torch.nn.Sequential(
        OrderedDict([
            ('good', torch.nn.Linear(2, 2)),
            ('bad', torch.nn.Linear(2, 1)),
        ]))
    return runner


class TestTrainRunnerNonfinite(unittest.TestCase):

    def test_forward_diagnostics_identify_predictions(self):
        runner = _runner()
        batch = {
            'states': torch.zeros(2, 32),
            'actions': torch.zeros(2, 50, 32),
            'prompt': ['test prompt'],
        }
        predictions = torch.zeros(2, 50, 28)
        predictions[0, 0, 0] = float('nan')

        details = runner._describe_forward_nonfinite(
            batch, {'predictions': predictions}, torch.tensor(float('nan')))

        self.assertIn('output.predictions', details)
        self.assertIn('nan=1', details)
        self.assertIn("prompt[0]='test prompt'", details)

    def test_gradient_diagnostics_identify_parameter(self):
        runner = _runner()
        for parameter in runner.vla.parameters():
            parameter.grad = torch.ones_like(parameter)
        runner.vla.bad.weight.grad[0, 0] = float('inf')

        details = runner._describe_gradient_nonfinite()

        self.assertIn('grad.bad.weight', details)
        self.assertIn('inf=1', details)

    def test_guard_reports_step_and_skips_optimizer(self):
        runner = _runner()

        with self.assertRaisesRegex(FloatingPointError, 'global step 10'):
            runner._raise_if_nonfinite(
                torch.tensor(float('nan')),
                'loss',
                details_factory=lambda: 'output.predictions contains NaN')


if __name__ == '__main__':
    unittest.main()

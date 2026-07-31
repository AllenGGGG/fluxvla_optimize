import unittest
from unittest import mock

import torch.distributed as dist

from fluxvla.engines.runners.fsdp_train_runner import FSDPTrainRunner


def _runner(device_id=3):
    runner = object.__new__(FSDPTrainRunner)
    runner.device_id = device_id
    return runner


class TestFSDPBarrier(unittest.TestCase):

    def test_nccl_barrier_uses_local_device(self):
        runner = _runner()

        with mock.patch.object(
                dist, 'get_backend', return_value=dist.Backend.NCCL), \
                mock.patch.object(dist, 'barrier') as barrier:
            runner._barrier()

        barrier.assert_called_once_with(device_ids=[3])

    def test_non_nccl_barrier_does_not_pass_device_ids(self):
        runner = _runner()

        with mock.patch.object(
                dist, 'get_backend', return_value=dist.Backend.GLOO), \
                mock.patch.object(dist, 'barrier') as barrier:
            runner._barrier()

        barrier.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()

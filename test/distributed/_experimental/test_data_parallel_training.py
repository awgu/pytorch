# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import itertools
import unittest
from typing import Iterable, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._experimental.data_parallel import data_parallel
from torch.distributed._tensor import distribute_tensor, DTensor, init_device_mesh
from torch.distributed._tensor.debug.comm_mode import CommDebugMode

from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    MLPStack,
    patch_all_gather,
    patch_reduce_scatter,
    test_compiled_fsdp,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)

c10d_ops = torch.ops.c10d
funcol = torch.ops.c10d_functional


class TestDataParallel(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_data_parallel_parity(self):
        model_args = ModelArgs(dropout_p=0.0)
        dp_mesh = init_device_mesh("cuda", (self.world_size,))
        model, optim, ref_model, ref_optim = self.init_transformer(
            model_args, dp_mesh, "replicate"
        )
        num_params = len(list(model.parameters())) + model_args.weight_tying

        inp = torch.randint(0, model_args.vocab_size, (3, 16), device="cuda")
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))

            with CommDebugMode() as fwd_comm_mode:
                loss = model(inp).sum()
            self.assertEqual(len(fwd_comm_mode.get_comm_counts()), 0)
            ref_loss = ref_model(inp).sum()
            self.assertEqual(loss, ref_loss)

            with CommDebugMode() as bwd_comm_mode:
                loss.backward()
            bwd_comm_counts = bwd_comm_mode.get_comm_counts()
            self.assertEqual(len(bwd_comm_counts), 1)
            self.assertEqual(bwd_comm_counts[funcol.all_reduce], num_params)
            ref_loss.backward()
            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

            optim.step()
            ref_optim.step()
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                self.check_tensor_parity(param, ref_param)

    @skip_if_lt_x_gpu(2)
    @parametrize("reduce_dtype", [None, torch.float32])
    def test_data_parallel_mp_parity(self, reduce_dtype: Optional[torch.dtype]):
        # NOTE: We cannot easily get numeric parity with weight tying. For
        # the ref model, we accumulate gradients across shared parameters in
        # bf16 and only at the end of backward cast to fp32 and all-reduce.
        # For the test model, we cast to fp32 and all-reduce once for all
        # instances of the weight-tied parameter.
        model_args = ModelArgs(dropout_p=0.0, weight_tying=False)
        dp_mesh = init_device_mesh("cuda", (self.world_size,))
        model, optim, ref_model, ref_optim = self.init_transformer(
            model_args, dp_mesh, "replicate", torch.bfloat16, reduce_dtype=reduce_dtype
        )
        ref_model_bf16 = copy.deepcopy(ref_model).to(torch.bfloat16)
        num_params = len(list(model.parameters())) + model_args.weight_tying

        inp = torch.randint(0, model_args.vocab_size, (3, 16), device="cuda")
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))

            with CommDebugMode() as fwd_comm_mode:
                loss = model(inp).sum()
            self.assertEqual(len(fwd_comm_mode.get_comm_counts()), 0)
            ref_loss = ref_model_bf16(inp).sum()
            self.assertEqual(loss, ref_loss)

            with CommDebugMode() as bwd_comm_mode:
                loss.backward()
            bwd_comm_counts = bwd_comm_mode.get_comm_counts()
            self.assertEqual(len(bwd_comm_counts), 1)
            self.assertEqual(bwd_comm_counts[funcol.all_reduce], num_params)
            ref_loss.backward()
            for param_bf16, param in zip(
                ref_model_bf16.parameters(), ref_model.parameters()
            ):
                if reduce_dtype is None:
                    dist.all_reduce(param_bf16.grad)
                    param.grad = param_bf16.grad.to(torch.float32).detach()
                    param.grad.detach().div_(self.world_size)
                else:
                    assert reduce_dtype == torch.float32, f"{reduce_dtype}"
                    param.grad = param_bf16.grad.to(torch.float32).detach()
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                param_bf16.grad = None
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                self.check_tensor_parity(param.grad, ref_param.grad)

            optim.step()
            ref_optim.step()
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                self.check_tensor_parity(param, ref_param)
            for param_bf16, param in zip(
                ref_model_bf16.parameters(), ref_model.parameters()
            ):
                param_bf16.detach().copy_(param)

    @skip_if_lt_x_gpu(2)
    def test_fully_sharded_data_parallel_parity(self):
        model_args = ModelArgs(dropout_p=0.0)
        dp_mesh = init_device_mesh("cuda", (self.world_size,))
        model, optim, ref_model, ref_optim = self.init_transformer(
            model_args, dp_mesh, "fully_shard"
        )
        model.compile()
        num_params = len(list(model.parameters())) + model_args.weight_tying

        inp = torch.randint(0, model_args.vocab_size, (3, 16), device="cuda")
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))

            with CommDebugMode() as fwd_comm_mode:
                loss = model(inp).sum()
            fwd_comm_counts = fwd_comm_mode.get_comm_counts()
            self.assertEqual(len(fwd_comm_counts), 1)
            self.assertEqual(fwd_comm_counts[funcol.all_gather_into_tensor], num_params)
            ref_loss = ref_model(inp).sum()
            self.assertEqual(loss, ref_loss)

            with CommDebugMode() as bwd_comm_mode:
                loss.backward()
            bwd_comm_counts = bwd_comm_mode.get_comm_counts()
            self.assertEqual(len(bwd_comm_counts), 2)
            self.assertEqual(bwd_comm_counts[funcol.all_gather_into_tensor], num_params)
            self.assertEqual(bwd_comm_counts[funcol.reduce_scatter_tensor], num_params)
            ref_loss.backward()

            for param in ref_model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            optim.step()
            ref_optim.step()
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                self.check_tensor_parity(param, ref_param)

    def init_transformer(
        self,
        model_args: ModelArgs,
        dp_mesh: DeviceMesh,
        dp_mode: str,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: Optional[torch.dtype] = None,
    ):
        torch.manual_seed(42)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).cuda()
        model = data_parallel(
            model, dp_mesh, dp_mode, param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        if model_args.weight_tying:
            model.output.parametrizations.weight.original = (
                model.tok_embeddings.parametrizations.weight.original
            )
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for ref_param, param in zip(ref_model.parameters(), model.parameters()):
            self.check_tensor_parity(param, ref_param)

        return model, optim, ref_model, ref_optim

    def check_tensor_parity(self, dtensor: DTensor, ref_tensor: torch.Tensor):
        ref_dtensor = distribute_tensor(
            ref_tensor, dtensor.device_mesh, dtensor.placements
        )
        self.assertEqual(ref_dtensor.to_local(), dtensor.to_local())


instantiate_parametrized_tests(TestDataParallel)

if __name__ == "__main__":
    run_tests()

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh

from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed._tensor.placement_types import _Partial
from torch.testing._internal.composite_compliance import is_view_fn

from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen, checkpoint


def _fsdp_recomp_policy():
    def _custom_policy(mode, func, *args, **kwargs):
        return not is_view_fn(func) and func not in {
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            torch.ops._c10d_functional.wait_tensor.default,
        }

    return _custom_policy


class ReplicateComputation(nn.Module):
    def __init__(self, device_mesh: DeviceMesh, param_sharding: Tuple[Placement, ...]):
        super().__init__()
        self.device_mesh = device_mesh
        self.param_sharding = param_sharding
        self.compute_placements = [Replicate()] * self.device_mesh.ndim
        self.grad_placements = [_Partial()] * self.device_mesh.ndim

    def forward(self, x: DTensor):
        return x.redistribute(placements=self.compute_placements).to_local(
            grad_placements=self.grad_placements
        )


class ToDtype(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        return x.to(self.dtype)


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "replicate",
    *,
    param_dtype: Optional[torch.dtype] = None,
    reduce_dtype: Optional[torch.dtype] = None,
):
    if mode == "replicate":
        param_sharding = (Replicate(),)
    elif mode == "fully_shard":
        param_sharding = (Shard(0),)
    elif mode == "hybrid_shard":
        # replicate inter-host, fully shard intra-host
        param_sharding = (Replicate(), Shard(0))
        assert (
            device_mesh.ndim == 2
        ), f"HSDP requires 2D DeviceMesh but got {device_mesh}"
    else:
        raise ValueError(f"Unsupported mode {mode}")

    if reduce_dtype is None:
        reduce_dtype = param_dtype
    modules_list = list(model.modules())
    for module in modules_list:
        named_params = dict(module.named_parameters(recurse=False))
        for param_name, param in named_params.items():
            module.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param, device_mesh, param_sharding)),
            )
            if param_dtype is not None and reduce_dtype == param_dtype:
                # Gradient reduction before cast back to fp32
                nn.utils.parametrize.register_parametrization(
                    module, param_name, ToDtype(param_dtype), unsafe=True
                )
            nn.utils.parametrize.register_parametrization(
                module,
                param_name,
                ReplicateComputation(device_mesh, param_sharding),
                unsafe=True,
            )
            if param_dtype is not None and reduce_dtype == param.dtype:
                # Cast back to fp32 before gradient reduction
                nn.utils.parametrize.register_parametrization(
                    module, param_name, ToDtype(param_dtype), unsafe=True
                )
    for param in model.parameters():
        param.register_hook(lambda grad: grad.div_(device_mesh.size()))

    if mode == "fully_shard" or mode == "hybrid_shard":

        def fsdp_policy():
            return _pt2_selective_checkpoint_context_fn_gen(_fsdp_recomp_policy())

        orig_forward = model.forward
        model.forward = lambda *args, **kwargs: checkpoint(
            orig_forward, *args, use_reentrant=False, context_fn=fsdp_policy, **kwargs
        )

    return model

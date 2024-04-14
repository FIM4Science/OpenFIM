import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from pkg_resources import packaging
from torch.distributed.fsdp import MixedPrecision


fp16 = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
    cast_forward_inputs=True,
    # cast_root_forward_inputs=True
)

bf16 = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)

bf16_mixed = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

fp16_mixed = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)


def is_bfloat_supported() -> bool:
    return (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )


precisions_types = {"fp16": fp16, "bf16": bf16, "bf16_mixed": bf16_mixed, "fp16_mixed": fp16_mixed, "fp32_policy": fp32_policy}

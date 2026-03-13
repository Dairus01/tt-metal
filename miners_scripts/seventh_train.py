#This is the miner whose training code scores seventh on the mining leaderboard, his MFU is 47.48%, 
#"mfu": 47.478027575963196,
#"tokens_per_second": 6483.670573642682,
#"total_tokens": 655360,
#"wall_time_seconds": 101.078546875,  here is his code:

import functools
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as _FlashCELoss
    _flash_ce = _FlashCELoss(ignore_index=-100)
    _USE_FLASH_CE = True
except ImportError:
    _USE_FLASH_CE = False

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float
    final_state: dict | None = None


_COMPILED = {}
_PREPARED = set()


def get_strategy():
    return "fsdp"


def _prepare_model(model):
    mid = id(model)
    if mid in _PREPARED:
        return
    _PREPARED.add(mid)
    if hasattr(model, "config"):
        model.config.use_cache = False
        try:
            model.config.output_hidden_states = False
        except Exception:
            pass
        try:
            model.config.output_attentions = False
        except Exception:
            pass


def _get_wrap_policy(model):
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        layer_cls = model.model.layers[0].__class__
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h") and len(model.transformer.h) > 0:
        layer_cls = model.transformer.h[0].__class__
    else:
        return None
    return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})


def _get_compiled_fwd(model):
    key = id(model)
    if key in _COMPILED:
        return _COMPILED[key]

    def fwd(input_ids):
        return model(input_ids).logits

    try:
        compiled = torch.compile(fwd, mode="reduce-overhead", dynamic=False)
    except Exception:
        compiled = fwd

    _COMPILED[key] = compiled
    return compiled


def inner_steps(model, data_iterator, optimizer, num_steps, device, num_gpus=1):
    _prepare_model(model)

    wrap_policy = _get_wrap_policy(model)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=mp_policy,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
        forward_prefetch=True,
    )

    compiled_fwd = _get_compiled_fwd(model)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            fused=True,
        )

    ce_fn = _flash_ce if _USE_FLASH_CE else None

    all_inputs = []
    all_labels = []
    tokens_per_batch = 0
    for _ in range(num_steps):
        batch = next(data_iterator).to(device, dtype=torch.long, non_blocking=True)
        all_inputs.append(batch[:, :-1].contiguous())
        all_labels.append(batch[:, 1:].contiguous())
        tokens_per_batch = batch.numel()

    torch.cuda.synchronize(device)

    total_tokens = num_steps * tokens_per_batch
    opt_step = optimizer.step
    opt_zero = optimizer.zero_grad

    for step in range(num_steps):
        logits = compiled_fwd(all_inputs[step])
        if ce_fn is not None:
            loss = ce_fn(logits.view(-1, logits.size(-1)), all_labels[step].view(-1))
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                all_labels[step].view(-1),
                ignore_index=-100,
            )
        loss.backward()
        opt_step()
        opt_zero(set_to_none=True)

    final_logits = logits.detach()
    final_loss = loss.item()

    full_state = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            sd = model.state_dict()
            if rank == 0:
                full_state = dict(sd)

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
        final_state=full_state,
    )

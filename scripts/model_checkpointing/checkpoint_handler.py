# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,  # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, cfg):
    # torch.manual_seed(103)
    folder_name = cfg.dist_checkpoint_root_folder + "/" + cfg.dist_checkpoint_folder + "-" + cfg.model_name

    load_dir = Path.cwd() / folder_name

    if not load_dir.exists():
        if rank == 0:
            print("No sharded_state_dict checkpoint directory found...skipping")
        return
    if rank == 0:
        print(f"loading model from model path: {load_dir} ")
    reader = FileSystemReader(load_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")

        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print("checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"])
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg, optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""

    folder_name = cfg.dist_checkpoint_root_folder + "/" + cfg.dist_checkpoint_folder + "-" + cfg.model_name

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(f"Checkpoint Time = {t1-t0:.4f}\n")


def save_model_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        print("--> saving model ...")
        # create save path
        # folder_name = cfg.dist_checkpoint_root_folder + "/" + cfg.dist_checkpoint_folder + "-" + cfg.model_name
        folder_name = "results/test/"
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = "gpt2-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)

        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")


def load_model_checkpoint(model, rank, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = Path(f"results/test/gpt2-{cfg}.pt")
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(f"model checkpoint {full_state_dict_model_path} not present. Returning...")
        return

    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    print("model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy, optim_state_dict_config=optim_save_policy)
    optim_state = FSDP.optim_state_dict(model, optimizer)

    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = "results/test/"
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = "optimizer" + "-" + "gpt2" + "-" + str(epoch) + ".pt"
        opt_save_full_path = save_dir / opt_save_name

        print("--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, cfg, rank, optim):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """
    optimizer_checkpoint_path = Path(f"results/test/optimizer-gpt2-{cfg}.pt")
    if not optimizer_checkpoint_path.is_file():
        print(f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. ")
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # # called from all ranks, though only rank0 has a valid param for full_osd
    _sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    # FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy, optim_save_policy)

    # optim_state_dict = FSDP.optim_state_dict_to_load(full_osd, model, optim)
    # optim.load_state_dict(optim_state_dict)

    print(f"optimizer shard loaded on rank {rank}")


def load_sharded_model_single_gpu(model, model_path):
    state_dict = {"model": model.state_dict()}

    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(model_path),
        no_dist=True,
    )

    model.load_state_dict(state_dict["model"])

    print(f"Sharded state checkpoint loaded from {model_path}")
    return model

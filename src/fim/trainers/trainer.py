import functools
import logging
import os
from datetime import datetime
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from ..data.dataloaders import BaseDataLoader
from ..models.blocks import AModel
from ..utils.helper import GenericConfig, create_optimizers, create_schedulers, verify_str_arg
from ..utils.logging import RankLoggerAdapter
from .checkpoint import TrainCheckpoint, TrainCheckpointFSDPFullStateDict, apply_fsdp_checkpointing
from .mixed_precision import fp16_mixed, is_bfloat_supported, precisions_types
from .utils import (
    GPUMemoryTrace,
    StepProgressBarFactory,
    TrainingTimePerformanceTracker,
    TrainLogging,
    TrainLossTracker,
)


class Trainer:
    def __init__(self, model: AModel, dataloader: BaseDataLoader, config: GenericConfig, resume: bool = False) -> None:
        self.config = config
        self.resume = resume
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self.dataloader: BaseDataLoader = dataloader
        self._setup_variables()
        self._prepare_experiment_dirs()
        self.training_logger = TrainLogging(self.experiment_dir, self.config.trainer.logging_format, self.rank)
        self.training_loss_tracker = TrainLossTracker()
        self.validation_loss_tracker = TrainLossTracker()
        self.max_steps = self.dataloader.n_train_batches * self.config.trainer.epochs // self.gradient_accumulation_steps
        self.steps_in_epoch = self.dataloader.n_train_batches // self.gradient_accumulation_steps
        self.schedulers: dict = create_schedulers(config.trainer.schedulers, self.max_steps, self.steps_in_epoch)
        self._prepare_model(model, config, resume)

        self._save_experiment_parameters()
        torch.autograd.set_detect_anomaly(self.config.trainer.detect_anomaly)

    def _prepare_model(self, model, config, resume):
        self.model = model
        checkpoint_class = TrainCheckpointFSDPFullStateDict if self.is_distributed else TrainCheckpoint
        self.optimizers = create_optimizers(self.model, config.optimizers)
        self.checkpointer = checkpoint_class(
            experiment_dir=self.experiment_dir,
            train_config=self.config,
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            training_logger=self.training_logger,
            train_loss_tracker=self.training_loss_tracker,
            validation_loss_tracker=self.validation_loss_tracker,
            grad_scaler=self.grad_scaler,
            rank=self.rank,
            is_peft=self.model.is_peft(),
        )
        if not self.is_distributed:
            self.model.to(config.experiment.device_map)
            if resume:
                self.start_epoch = self.checkpointer.load_checkpoint("last-epoch")
                self.optimizers = self.checkpointer.optimizers
        else:
            if resume:
                self.start_epoch = self.checkpointer.load_model_state("last-epoch")
                self.model = self.checkpointer.model

            self._fsdp_initialize()

            self.optimizers: dict = create_optimizers(self.model, config.optimizers)
            self.checkpointer.model = self.model
            self.checkpointer.optimizers = self.optimizers

            if resume:
                self.checkpointer.load_optimizers_state("last-epoch")
                self.optimizers = self.checkpointer.optimizers
            dist.barrier()

    def _setup_variables(self):
        self.rank = 0
        self.local_rank = 0 if torch.cuda.is_available() and self.config.experiment.device_map else "cpu"
        if self.is_distributed:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])

        self.number_of_debug_iterations = self.config.trainer.debug_iterations
        self.debug_mode = self.number_of_debug_iterations is not None
        self.start_epoch: int = 0
        self.n_epochs: int = self.config.trainer.epochs if not self.debug_mode else 2
        self.gradient_accumulation_steps: int = self.config.trainer.gradient_accumulation_steps
        self.best_metric: str = self.config.trainer.best_metric
        self.precision = self.config.trainer.precision
        self._use_mixeprecision = self.precision is not None  # and not self.is_distributed
        self._auto_cast_type = torch.float16
        if is_bfloat_supported() and self._use_mixeprecision:
            if self.rank == 0:
                self.logger.warning("MIXED_PRECISION: There is bfloat16 support on your system. bfloat16 will be used instead of float16!")
            self._auto_cast_type = torch.bfloat16
        self.grad_scaler = None
        if self._use_mixeprecision and not self.is_distributed:
            self.grad_scaler = torch.cuda.amp.GradScaler()
        elif self._use_mixeprecision and self.is_distributed:
            self.grad_scaler = ShardedGradScaler()

    def _fsdp_initialize(self):
        wrap_policy = self._get_wrap_policy(self.model)
        sharding_strategy = self._get_sharding_strategy()
        mixed_precision_policy = self._get_mixed_precision_policy()
        self.model = FSDP(
            self.model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=self.resume,
            use_orig_params=True,  # self.model.is_peft(),
            # ignored_modules=[self.model.fim_base], # ignored_states
        )
        if self.config.distributed.activation_chekpoint:
            activation_check_fn = self.model.fsdp_activation_check_fn()
            apply_fsdp_checkpointing(self.model, activation_check_fn)
        if self.rank == 0:
            self.logger.info(self.model)

    def _get_mixed_precision_policy(self):
        mixed_precision = None
        _precision = self.config.trainer.precision
        if _precision in ["bf16_mixed", "bf16"] and is_bfloat_supported():
            self.logger.info("Using precision policy %s!", _precision)
            mixed_precision = precisions_types[_precision]
        elif _precision == "bf16_mixed" and not is_bfloat_supported():
            self.logger.warning("bf16_mixed is not supported. Using fp16_mixed!")
            mixed_precision = fp16_mixed
        elif _precision == "bf16" and not is_bfloat_supported():
            self.logger.warning("bf16 is not supported. Using fp16!")
            mixed_precision = precisions_types["fp16"]
        elif _precision in precisions_types.keys():
            mixed_precision = precisions_types[_precision]

        return mixed_precision

    def _get_wrap_policy(self, model: BaseDataLoader):
        match self.config.distributed.wrap_policy:
            case "MODEL_SPECIFIC":
                return model.get_fsdp_policy(int(float(self.config.distributed.min_num_params)))
            case "SIZE_BASED":
                return functools.partial(size_based_auto_wrap_policy, min_num_params=int(float(self.config.distributed.min_num_params)))
            case _:
                return None

    def _prepare_experiment_dirs(self) -> None:
        name = self.config.experiment.name
        if self.resume:
            # remove "-seed-n" at end
            name = name[: name.rfind("-experiment-seed-")]
            date = ""
        elif self.config.experiment.name_add_date:
            # add date as string if wanted
            date = "_" + datetime.now().strftime("%m-%d-%H%M")
        else:
            date = ""
        name = name + date

        if len(name) > 200:
            name = "_".join([i if i.isdigit() else i[0:3] for i in name.split("_")])
        # start_time = datetime.now().strftime("%d%m%y_%H%M%S")
        self.experiment_dir = Path(self.config.trainer.experiment_dir) / name

        if self.rank == 0:
            self.logger.info("Preparing Experiment Directory ...")
            self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def _save_experiment_parameters(self):
        if self.rank != 0:
            return
        params_path = self.experiment_dir / "train_parameters.yaml"
        self.logger.info("Saving training configuration to %s", params_path)
        yaml.dump(self.config.to_dict(), open(params_path, "w", encoding="utf-8"), default_flow_style=False)

    def train(self):
        p_bar = StepProgressBarFactory.create_epoch_progress_bar(self.n_epochs, self.rank, self.start_epoch)

        time_trace = TrainingTimePerformanceTracker(self.rank)
        for epoch in range(self.start_epoch, self.n_epochs):
            if torch.cuda.is_available() and self.config.experiment.device_map != "cpu":
                mem_trace = GPUMemoryTrace(self.rank)

            # train step
            time_trace.start_epoch()
            train_epoch_stats = self._train_epoch(epoch)
            time_trace.stop_epoch()

            # validation step
            time_trace.start_timer("Validation")
            validation_epoch_stats = self._validation_epoch(epoch)
            time_trace.stop_timer("Validation")

            # handle & save current checkpoint
            self._update_learning_rates("epoch")
            self.training_logger.log_epoch(epoch, train_epoch_stats, validation_epoch_stats)
            time_trace.start_timer("Checkpoint")
            self.checkpointer.save_checkpoint(epoch, train_epoch_stats, validation_epoch_stats)
            time_trace.stop_timer("Checkpoint")
            p_bar.update_and_set_postfix(1, train_epoch_stats, validation_epoch_stats, ["loss", "ppl"])
            if torch.cuda.is_available() and self.config.experiment.device_map != "cpu":
                mem_trace.print_summary()
            time_trace.print_elapsed_time("Epoch")
            time_trace.print_elapsed_time("Validation")
            time_trace.print_elapsed_time("Checkpoint")
            time_trace.print_epochs_time_statistics()

        self.training_logger.clear_logging_resources()
        p_bar.close()
        return None

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        p_bar = StepProgressBarFactory.create_train_progress_bar(self.dataloader.n_train_batches, self.rank)
        data_it = self.dataloader.train_it
        if self.debug_mode:
            data_it = islice(data_it, self.number_of_debug_iterations)
        step = epoch * self.steps_in_epoch
        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(self.training_logger.tensorboard_dir),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        # ) as prof:
        for batch_idx, batch in enumerate(data_it):
            # prof.step()
            step = (step + batch_idx) // self.gradient_accumulation_steps
            batch_stats = self._train_batch(step, batch)
            self.training_loss_tracker.add_batch_stats(batch_stats)
            p_bar.update_and_set_postfix(1, batch_stats["losses"])
            self.training_logger.log_train_batch(epoch, batch_idx, batch_stats)
        self.training_loss_tracker.summarize_epoch()
        p_bar.close()
        del p_bar
        return self.training_loss_tracker.get_last_epoch_stats()

    def _train_batch(self, step: int, batch: dict) -> dict:
        self._move_batch_to_local_rank(batch)
        with torch.amp.autocast("cuda", enabled=self._use_mixeprecision and torch.cuda.is_available(), dtype=self._auto_cast_type):
            stats = self.model(batch, training=True, schedulers=self.schedulers, step=step)
            losses = stats["losses"]
            loss = losses["loss"]
            loss = loss / self.gradient_accumulation_steps
        losses = {k: v.detach().float() for k, v in losses.items()}
        histograms = {}
        # if "histograms" in stats:
        #     histograms = {k: v.detach().float() for k, v in stats["histograms"].items()}

        lrs = self._model_update_step(step, loss)
        return {"losses": losses | lrs, "histograms": histograms, "line_plots": stats.get("visualizations", {})}
        # return {"losses": losses | lrs}

    def _model_update_step(self, step: int, loss: torch.Tensor):
        lrs = {}
        update_gradients = ((step + 1) % self.gradient_accumulation_steps == 0) or ((step + 1) == self.dataloader.n_train_batches)
        if self.precision or self.config.model.use_bf16:
            self.grad_scaler.scale(loss).backward()
            if update_gradients:
                for opt_name, optimizer in self.optimizers.items():
                    self.grad_scaler.step(optimizer["opt"])
                    self.grad_scaler.update()
                    optimizer["opt"].zero_grad()
                    lrs[f"OPT-{opt_name.upper()}-LR"] = torch.tensor(
                        optimizer["opt"].get_last_lr()[0]
                        if hasattr(optimizer["opt"], "get_last_lr")
                        else optimizer["opt"].param_groups[0]["lr"],
                        device=self.local_rank,
                    )

        else:
            loss.backward()
            if update_gradients:
                for opt_name, optimizer in self.optimizers.items():
                    optimizer["opt"].step()
                    optimizer["opt"].zero_grad()
                    lrs[f"OPT-{opt_name.upper()}-LR"] = torch.tensor(
                        optimizer["opt"].get_last_lr()[0]
                        if hasattr(optimizer["opt"], "get_last_lr")
                        else optimizer["opt"].param_groups[0]["lr"],
                        device=self.local_rank,
                    )
        if update_gradients:
            self._update_learning_rates("minibatch")
        return lrs

    def _move_batch_to_local_rank(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.local_rank)

    def _validation_epoch(self, epoch: int) -> dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = StepProgressBarFactory.create_validation_progress_bar(self.dataloader.n_validation_batches, self.rank)
            data_it = self.dataloader.validation_it
            if self.debug_mode:
                data_it = islice(data_it, self.number_of_debug_iterations)
            for batch_idx, batch in enumerate(data_it):
                batch_stats = self._validation_batch(batch_idx, batch)
                self.validation_loss_tracker.add_batch_losses(batch_stats.get("losses"))
                # self.validation_loss_tracker.add_batch_histograms(batch_stats.get("histograms"))
                self.validation_loss_tracker.add_batch_line_plots(batch_stats.get("line_plots"))
                p_bar.update_and_set_postfix(1, batch_stats["losses"])
            self.validation_loss_tracker.summarize_epoch()

            p_bar.close()
            del p_bar
        return self.validation_loss_tracker.get_last_epoch_stats()

    def _validation_batch(self, step: int, batch: dict) -> dict:
        self._move_batch_to_local_rank(batch)
        stats = self.model(batch, training=False)
        losses = {k: v.detach().float() for k, v in stats["losses"].items()}
        histograms = {}
        # if "histograms" in stats:
        #     histograms = {k: v.detach().float() for k, v in stats["histograms"].items()}
        return {"losses": losses, "histograms": histograms, "line_plots": stats.get("visualizations", {})}
        # return {"losses": losses}

    def _update_learning_rates(self, call_place: str):
        verify_str_arg(call_place, "call_place", ["epoch", "minibatch"])
        for k, v in self.optimizers.items():
            if v["schedulers"] is not None:
                for step_type, scheduler in v["schedulers"]:
                    if step_type == call_place:
                        self.training_logger.file_logger.info("Updating the leargning rate of '%s'", k)
                        self.logger.info("Updating the leargning rate of '%s'", k)
                        scheduler.step()

    def _get_sharding_strategy(self) -> ShardingStrategy:
        sharding_strategy = self.config.distributed.sharding_strategy
        if sharding_strategy.upper() == "FULL_SHARD":
            return ShardingStrategy.FULL_SHARD
        elif sharding_strategy.upper() == "SHARD_GRAD_OP":
            return ShardingStrategy.SHARD_GRAD_OP
        elif sharding_strategy.upper() == "NO_SHARD":
            return ShardingStrategy.NO_SHARD
        else:
            raise ValueError(f"Sharding strategy {sharding_strategy.upper()} not supported!")

    @property
    def is_distributed(self) -> bool:
        return self.config.distributed.enabled

    def __str__(self) -> str:
        return f"Trainer:\n\n\tModel: {self.model}\n\n\tDataloader: {self.dataloader}\n\n\tOptimizers: {self.optimizers}\n\n\tSchedulers: {self.schedulers}"


class TrainerFactory:
    trainer_types = {}

    @classmethod
    def register(cls, trainer_type: str, trainer_class: Trainer):
        cls.trainer_types[trainer_type] = trainer_class

    @classmethod
    def create(cls, name: str, **kwargs) -> Trainer:
        trainer_class = cls.trainer_types.get(name)
        if trainer_class:
            return trainer_class(**kwargs)
        else:
            raise ValueError("Invalid trainer type!")


TrainerFactory.register("Trainer", Trainer)

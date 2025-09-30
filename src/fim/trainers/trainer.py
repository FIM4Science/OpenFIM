import functools
import itertools
import logging
import os
import random
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import IterableDataset

from fim.trainers.evaluation_epochs import EvaluationEpoch

from ..data.dataloaders import BaseDataLoader
from ..models.blocks import AModel
from ..utils.helper import GenericConfig, create_class_instance, create_optimizers, create_schedulers, verify_str_arg
from ..utils.logging import RankLoggerAdapter
from .checkpoint import TrainCheckpoint, TrainCheckpointFSDPFullStateDict, apply_fsdp_checkpointing
from .mixed_precision import fp16_mixed, is_bfloat_supported, precisions_types
from .utils import (
    GPUMemoryTrace,
    StepProgressBarFactory,
    TrainingTimePerformanceTracker,
    TrainLogging,
    TrainLossTracker,
    get_accel_type,
    is_accelerator_available,
    move_batch_to_local_rank,
)


class Trainer:
    def __init__(
        self,
        model: AModel,
        dataloader: BaseDataLoader,
        config: GenericConfig,
        resume: Optional[Path | str] = None,
        resume_epoch: Optional[str] = "last-epoch",
    ) -> None:
        self.config = config
        self.resume = resume
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self.dataloader: BaseDataLoader = dataloader
        self._setup_variables()
        self._prepare_experiment_dirs()
        self.training_logger = TrainLogging(self.experiment_dir, self.config.trainer.logging_format, self.rank)
        self.training_loss_tracker = TrainLossTracker()
        self.validation_loss_tracker = TrainLossTracker()
        self.evaluation_loss_tracker = TrainLossTracker()
        self.validation_epoch_frequency = (
            self.config.trainer.validation_epoch_frequency if hasattr(self.config.trainer, "validation_epoch_frequency") else 1
        )
        self.max_steps = self.dataloader.n_train_batches * self.config.trainer.epochs // self.gradient_accumulation_steps
        self.steps_in_epoch = self.dataloader.n_train_batches // self.gradient_accumulation_steps
        self.schedulers: dict = create_schedulers(config.trainer.schedulers, self.max_steps, self.steps_in_epoch)
        self._prepare_model(model, config, resume, resume_epoch)
        self._evaluation_epoch: EvaluationEpoch = self._prepare_evaluation_epoch()

        self._save_experiment_parameters()
        torch.autograd.set_detect_anomaly(self.config.trainer.detect_anomaly)

        self.profiler_config: dict = self.config.trainer.to_dict().get("profiler")  # None or {"trace_path": , "num_batches": }

    def _prepare_model(self, model, config, resume: Optional[Path | str] = None, resume_epoch: Optional[str] = "last_epoch"):
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
            evaluation_loss_tracker=self.evaluation_loss_tracker,
            grad_scaler=self.grad_scaler,
            rank=self.rank,
            is_peft=False,  # self.model.is_peft(),
            resume_dir=resume,
            hub_model_id=config.trainer.get("hub_model_id", None),
        )
        if config.experiment.device_map == "auto":
            device = torch.device(self.accel_type)
        else:
            device = torch.device(config.experiment.device_map)

        if not self.is_distributed:
            self.model.to(device)
            if resume is not None:
                self.start_epoch = self.checkpointer.load_checkpoint(resume_epoch)
                self.optimizers = self.checkpointer.optimizers
        else:
            if resume is not None:
                self.start_epoch = self.checkpointer.load_model_state(resume_epoch)
                self.model = self.checkpointer.model

            self._fsdp_initialize()

            self.optimizers: dict = create_optimizers(self.model, config.optimizers)
            self.checkpointer.model = self.model
            # self.checkpointer.optimizers = self.optimizers

            # if resume is not None:
            # self.checkpointer.load_optimizers_state("last-epoch")
            # self.optimizers = self.checkpointer.optimizers

    def _setup_variables(self):
        self.is_accelerator = is_accelerator_available()
        self.accel_type = get_accel_type()
        self.rank = 0
        self.local_rank = 0 if self.is_accelerator and self.config.experiment.device_map != "cpu" else "cpu"
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
        self._use_mixeprecision = (self.precision is not None) and (self.precision != "fp32_policy")
        if self._use_mixeprecision:
            self._auto_cast_type = torch.float16
            if is_bfloat_supported():
                if self.rank == 0:
                    self.logger.warning(
                        "MIXED_PRECISION: There is bfloat16 support on your system. bfloat16 will be used instead of float16!"
                    )
                self._auto_cast_type = torch.bfloat16
        else:
            self._auto_cast_type = torch.float32
        self.grad_scaler = None
        if self._use_mixeprecision and not self.is_distributed:
            self.grad_scaler = torch.amp.GradScaler(self.accel_type)
        elif self._use_mixeprecision and self.is_distributed:
            self.grad_scaler = ShardedGradScaler(self.accel_type)

    def _fsdp_initialize(self):
        # Ensure each rank uses its own CUDA device before wrapping the model
        if self.config.experiment.device_map != "cpu":
            try:
                torch.cuda.set_device(self.local_rank)
            except Exception:
                pass
        wrap_policy = self._get_wrap_policy(self.model)
        sharding_strategy = self._get_sharding_strategy()
        mixed_precision_policy = self._get_mixed_precision_policy()
        self.model = FSDP(
            self.model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mixed_precision_policy,
            device_id=(self.local_rank if self.config.experiment.device_map != "cpu" else None),
            limit_all_gathers=True,
            sync_module_states=self.resume is not None,
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

        if self.config.experiment.name_add_date:
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
        model_architecture_path = self.experiment_dir / "model_architecture.txt"
        # For iterable datasets and map-style datasets alike, prefer taking a collated batch
        # directly from the DataLoader so that any custom collate logic is applied.
        try:
            if isinstance(self.dataloader.train_it.dataset, IterableDataset):
                x = next(iter(self.dataloader.train_it))
            elif isinstance(self.dataloader.train_it.dataset[0], tuple):
                logging.warning("Saving model architecture is not supported for tuple datasets!")
                return
            else:
                x = next(iter(self.dataloader.train_it))
        except StopIteration:
            # Can occur with IterableDatasets when per-worker caps allocate zero batches to some workers
            self.logger.warning("Skipping model architecture summary: no sample batch available at startup.")
            return
        # NOTE: This block is ONLY used for writing the model architecture summary at startup.
        # If the collate returns a grouped batch (keys are ints, e.g. different mark dimensions),
        # we pick one subgroup to form a valid example batch for summary. Training is unaffected.
        if isinstance(next(iter(x.keys())), int):  # Catch case where we use groupings
            # Select a random value of x
            x = x[random.choice(list(x.keys()))]
        x = {key: val.to(self.model.device) for key, val in x.items()}

        with open(model_architecture_path, "w") as f:
            f.write(str(self.model.summary(x)))
            self.logger.info("Saving model architecture to %s", model_architecture_path)

    def _prepare_evaluation_epoch(self) -> EvaluationEpoch:
        if hasattr(self.config.trainer, "evaluation_epoch"):
            evaluation_epoch_kwargs: dict = self.config.trainer.evaluation_epoch.to_dict()
            evaluation_epoch_path = evaluation_epoch_kwargs.pop("path")
            _evaluation_epoch: EvaluationEpoch = create_class_instance(
                evaluation_epoch_path,
                evaluation_epoch_kwargs,
                self.model,
                self.dataloader,
                self.evaluation_loss_tracker,
                self.debug_mode,
                self.local_rank,
                self.accel_type,
                self._use_mixeprecision,
                self.is_accelerator,
                self._auto_cast_type,
            )

        else:

            def _evaluation_epoch(epoch):
                return {}

        return _evaluation_epoch

    def train(self):
        p_bar = StepProgressBarFactory.create_epoch_progress_bar(self.n_epochs, self.rank, self.start_epoch)

        time_trace = TrainingTimePerformanceTracker(self.rank)
        for epoch in range(self.start_epoch, self.n_epochs):
            if self.accel_type == "cuda":
                mem_trace = GPUMemoryTrace(self.rank)

            # train step
            time_trace.start_epoch()
            train_epoch_stats = self._train_epoch(epoch)
            time_trace.stop_epoch()

            # validation step
            if epoch % self.validation_epoch_frequency == 0:
                time_trace.start_timer("Validation")
                validation_epoch_stats = self._validation_epoch(epoch)
                time_trace.stop_timer("Validation")
            else:
                validation_epoch_stats = train_epoch_stats

            # evaluation step
            time_trace.start_timer("Evaluation")
            evaluation_epoch_stats = self._evaluation_epoch(epoch)
            time_trace.stop_timer("Evaluation")

            # handle & save current checkpoint
            self._update_learning_rates("epoch")
            # On non-zero ranks, skip checkpoint bookkeeping when there are no batches (e.g., empty per-worker cap)
            if (train_epoch_stats is None) or ("losses" not in train_epoch_stats):
                train_epoch_stats = {"losses": {"loss": float("inf")}}
            if (validation_epoch_stats is None) or ("losses" not in validation_epoch_stats):
                validation_epoch_stats = {"losses": {"loss": float("inf")}}
            self.training_logger.log_epoch(epoch, train_epoch_stats, validation_epoch_stats, evaluation_epoch_stats)
            time_trace.start_timer("Checkpoint")
            self.checkpointer.save_checkpoint(epoch, train_epoch_stats, validation_epoch_stats)
            time_trace.stop_timer("Checkpoint")
            p_bar.update_and_set_postfix(1, train_epoch_stats["losses"], validation_epoch_stats["losses"], ["loss", "ppl"])
            if self.accel_type == "cuda":
                mem_trace.print_summary()
            time_trace.print_elapsed_time("Epoch")
            time_trace.print_elapsed_time("Validation")
            time_trace.print_elapsed_time("Evaluation")
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
        if self.is_distributed and "train" in self.dataloader.samplers.keys():
            # Some samplers are proxies for IterableDatasets and only expose set_epoch
            sampler = self.dataloader.samplers["train"]
            if hasattr(sampler, "set_epoch") and callable(getattr(sampler, "set_epoch")):
                sampler.set_epoch(epoch)
        step = epoch * self.steps_in_epoch

        if self.profiler_config is not None:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], profile_memory=True
            )

            if (num_profiling_batches := self.profiler_config.get("num_batches")) is not None:
                data_it = itertools.islice(data_it, num_profiling_batches)

            prof.start()

        for batch_idx, batch in enumerate(data_it):
            if self.profiler_config is not None:
                prof.step()

            step_ = (step + batch_idx) // self.gradient_accumulation_steps
            with torch.profiler.record_function("train_batch"):
                batch_stats = self._train_batch(step_, batch)

                batch_stats = torch.utils._pytree.tree_map(
                    lambda x: x.to("cpu").detach() if isinstance(x, torch.Tensor) else x, batch_stats
                )

            with torch.profiler.record_function("batch_stats"):
                self.training_loss_tracker.add_batch_stats(batch_stats)

            # Update progress only on rank 0 to avoid cross-rank contention
            if self.rank == 0:
                p_bar.update_and_set_postfix(1, batch_stats["losses"])

            # Log only on rank 0 for clearer progress feedback
            if self.rank == 0:
                self.training_logger.log_train_batch(epoch, batch_idx, batch_stats)

            del batch
            del batch_stats

        # If no batches were processed (e.g., this rank/worker had zero), return None
        try:
            self.training_loss_tracker.summarize_epoch()
            last_stats = self.training_loss_tracker.get_last_epoch_stats()
        except Exception:
            last_stats = None
        p_bar.close()
        del p_bar

        if self.profiler_config is not None:
            prof.stop()
            prof.export_chrome_trace(self.profiler_config["trace_path"])

        return last_stats

    def _train_batch(self, step: int, batch: dict) -> dict:
        with torch.profiler.record_function("move_data_to_gpu"):
            batch = move_batch_to_local_rank(batch, self.local_rank)
            # Early-exit: if this rank received an empty/None batch, skip update to keep DDP in sync
            if batch is None:
                return {"losses": {"loss": torch.tensor(0.0, device=self.local_rank if self.local_rank != "cpu" else "cpu")}}
        with torch.profiler.record_function("forward_pass"):
            with torch.amp.autocast(
                self.accel_type,
                enabled=self._use_mixeprecision and self.is_accelerator,
                dtype=self._auto_cast_type,
            ):
                stats = self.model(batch, schedulers=self.schedulers, step=step)
                losses = stats["losses"]
                loss = losses["loss"]
                loss = loss / self.gradient_accumulation_steps

        with torch.profiler.record_function("backward_pass"):
            lrs = self._model_update_step(step, loss)

        # Keep tensors on device for distributed reductions (NCCL cannot all_reduce CPU tensors)
        if self.local_rank == "cpu" or self.config.experiment.device_map == "cpu":
            device_for_losses = torch.device("cpu")
        else:
            device_for_losses = torch.device("cuda", int(self.local_rank))
        losses = {k: (v.detach().float().to(device_for_losses) if isinstance(v, torch.Tensor) else v) for k, v in losses.items()}
        histograms = {}
        # if "histograms" in stats:
        #     histograms = {k: v.detach().float() for k, v in stats["histograms"].items()}
        return {"losses": losses | lrs, "histograms": histograms, "line_plots": stats.get("visualizations", {})}
        # return {"losses": losses | lrs}

    def _model_update_step(self, step: int, loss: torch.Tensor):
        lrs = {}
        update_gradients = ((step + 1) % self.gradient_accumulation_steps == 0) or ((step + 1) == self.dataloader.n_train_batches)
        if self._use_mixeprecision:
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
                if self.rank == 0:
                    p_bar.update_and_set_postfix(1, batch_stats["losses"])
            self.validation_loss_tracker.summarize_epoch()

            p_bar.close()
            del p_bar
        return self.validation_loss_tracker.get_last_epoch_stats()

    def _validation_batch(self, step: int, batch: dict) -> dict:
        batch = move_batch_to_local_rank(batch, self.local_rank)
        with torch.amp.autocast(
            self.accel_type,
            enabled=True,  # self._use_mixeprecision and self.is_accelerator,
            dtype=self._auto_cast_type,
        ):
            stats = self.model(batch)
        if self.local_rank == "cpu" or self.config.experiment.device_map == "cpu":
            device_for_losses = torch.device("cpu")
        else:
            device_for_losses = torch.device("cuda", int(self.local_rank))
        losses = {k: (v.detach().float().to(device_for_losses) if isinstance(v, torch.Tensor) else v) for k, v in stats["losses"].items()}
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
                        # self.training_logger.file_logger.info("Updating the leargning rate of '%s'", k)
                        # self.logger.info("Updating the leargning rate of '%s'", k)
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

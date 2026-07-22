#!/usr/bin/env python3
"""
Generic Hydra/OmegaConf hyperparameter tuning for SGG-Benchmark.

Place this file in:
    SGG-Benchmark/tools/tune_sgg_hydra.py

The original YAML is never modified. Each Ray Tune trial:
1. reloads the YAML;
2. applies FIXED_OVERRIDES;
3. applies values sampled from SEARCH_SPACE;
4. trains for the configured tuning budget;
5. validates and reports one objective score to Ray Tune.

Edit only SEARCH_SPACE and FIXED_OVERRIDES to add/remove parameters.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

# Make the repository importable when this script is launched from tools/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SGG-Benchmark asks for this import before the other project modules.
from sgg_benchmark.utils.env import setup_environment  # noqa: F401,E402

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from optuna.samplers import TPESampler

import ray
from ray import train, tune
from ray.tune.logger import (
    CSVLoggerCallback,
    JsonLoggerCallback,
    TBXLoggerCallback,
)
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

try:
    # Current Ray versions.
    from ray.tune import RunConfig
except ImportError:
    # Compatibility with older Ray 2.x versions.
    from ray.air import RunConfig

from sgg_benchmark.config import (  # noqa: E402
    load_config_from_file,
    update_config_from_list,
)
from sgg_benchmark.data import make_data_loader  # noqa: E402
from sgg_benchmark.engine.trainer import (  # noqa: E402
    assert_mode,
    get_mode,
    run_val,
    train_one_epoch,
)
from sgg_benchmark.modeling.detector import build_detection_model  # noqa: E402
from sgg_benchmark.solver import make_lr_scheduler, make_optimizer  # noqa: E402
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer  # noqa: E402
from sgg_benchmark.utils.logger import setup_logger  # noqa: E402
from sgg_benchmark.utils.miscellaneous import mkdir, set_seed  # noqa: E402


# =============================================================================
# USER-EDITABLE SECTION
# =============================================================================
#
# Supported distribution types:
#
# choice:
#   {"type": "choice", "values": [value1, value2, ...]}
#
# uniform:
#   {"type": "uniform", "low": 0.0, "high": 1.0}
#
# loguniform:
#   {"type": "loguniform", "low": 1e-5, "high": 1e-2}
#
# randint:
#   {"type": "randint", "low": 1, "high": 10}
#   "high" is exclusive.
#
# quniform / qloguniform:
#   {"type": "quniform", "low": 0.1, "high": 0.9, "q": 0.1}
#
# qrandint:
#   {"type": "qrandint", "low": 32, "high": 1025, "q": 32}
#
# Paths are OmegaConf dot paths and are case-insensitive in this script.
# Every path is checked before Ray starts.
#
SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    # Optimizer/training parameters
    "solver.base_lr": {
        "type": "loguniform",
        "low": 1.0e-5,
        "high": 5.0e-4,
    },
    "solver.ims_per_batch": {
        "type": "choice",
        "values": [2, 4, 8],
    },
    "solver.accum_steps": {
        "type": "choice",
        "values": [1, 2, 4],
    },

    # Relation-head parameters
    "model.roi_relation_head.batch_size_per_image": {
        "type": "choice",
        "values": [256, 512, 768, 1024],
    },
    "model.roi_relation_head.positive_fraction": {
        "type": "uniform",
        "low": 0.20,
        "high": 0.50,
    },
    "model.roi_relation_head.context_dropout_rate": {
        "type": "uniform",
        "low": 0.05,
        "high": 0.40,
    },
    "model.roi_relation_head.loss.logit_adjustment_tau": {
        "type": "uniform",
        "low": 0.10,
        "high": 1.00,
    },

    # Detector/post-processing examples.
    # Verify that the selected detector/meta-architecture actually reads
    # every parameter you decide to tune.
    "model.backbone.nms_thresh": {
        "type": "loguniform",
        "low": 1.0e-4,
        "high": 5.0e-2,
    },
    "model.roi_heads.nms": {
        "type": "uniform",
        "low": 0.30,
        "high": 0.70,
    },
    "model.roi_heads.fg_iou_threshold": {
        "type": "uniform",
        "low": 0.25,
        "high": 0.55,
    },
    "model.roi_heads.detections_per_img": {
        "type": "choice",
        "values": [50, 75, 100, 150],
    },
    "test.detections_per_img": {
        "type": "choice",
        "values": [50, 75, 100, 150],
    },
}

# Values always applied before the sampled overrides.
# Use this for parameters that must remain fixed during all trials.
FIXED_OVERRIDES: Dict[str, Any] = {
    # REACT++ normally uses AdamW.
    "solver.optimizer": "ADAMW",

    # Examples:
    # "dtype": "float16",
    # "test.allow_load_from_cache": False,
    # "model.roi_relation_head.max_pairs_inference": 300,
}

# Aliases supported by --metric.
METRIC_SUFFIXES: Dict[str, str] = {
    "mR": "_mean_recall",
    "R": "_recall",
    "F1": "_f1_score",
    "zR": "_zeroshot_recall",
    "ng-zR": "_ng_zeroshot_recall",
    "ng-R": "_recall_nogc",
    "ng-mR": "_ng_mean_recall",
}


# =============================================================================
# SEARCH-SPACE AND CONFIG UTILITIES
# =============================================================================

_MISSING = object()


def normalize_path(path: str) -> str:
    """Convert a YACS-style or Hydra-style path to lowercase Hydra notation."""
    return path.strip().lower()


def make_domain(spec: Mapping[str, Any]):
    """Convert a declarative SEARCH_SPACE entry to a Ray Tune domain."""
    kind = str(spec["type"]).lower()

    if kind == "choice":
        values = list(spec["values"])
        if not values:
            raise ValueError("choice requires a non-empty 'values' list")
        return tune.choice(values)

    if kind == "uniform":
        return tune.uniform(float(spec["low"]), float(spec["high"]))

    if kind == "loguniform":
        return tune.loguniform(float(spec["low"]), float(spec["high"]))

    if kind == "randint":
        return tune.randint(int(spec["low"]), int(spec["high"]))

    if kind == "quniform":
        return tune.quniform(
            float(spec["low"]),
            float(spec["high"]),
            float(spec["q"]),
        )

    if kind == "qloguniform":
        return tune.qloguniform(
            float(spec["low"]),
            float(spec["high"]),
            float(spec["q"]),
        )

    if kind == "qrandint":
        return tune.qrandint(
            int(spec["low"]),
            int(spec["high"]),
            int(spec["q"]),
        )

    raise ValueError(
        f"Distribution type '{kind}' is not supported. "
        "Use choice, uniform, loguniform, randint, quniform, "
        "qloguniform or qrandint."
    )


def ensure_path_exists(cfg: DictConfig, path: str) -> Any:
    """Return the current value and fail clearly when a config path is invalid."""
    normalized = normalize_path(path)
    value = OmegaConf.select(cfg, normalized, default=_MISSING)

    if value is _MISSING:
        raise KeyError(
            f"Config path '{path}' does not exist in the merged Hydra config."
        )

    return value


def apply_overrides(
    cfg: DictConfig,
    overrides: Mapping[str, Any],
    *,
    validate: bool = True,
) -> DictConfig:
    """Apply flat dot-path overrides to an OmegaConf DictConfig."""
    for raw_path, value in overrides.items():
        path = normalize_path(raw_path)

        if validate:
            ensure_path_exists(cfg, path)

        OmegaConf.update(cfg, path, value, merge=False)

    return cfg


def set_task_mode(cfg: DictConfig, task: str) -> None:
    """Set the flags used by SGG-Benchmark for sgdet, sgcls or predcls."""
    with open_dict(cfg):
        if task == "sgdet":
            cfg.model.roi_relation_head.use_gt_box = False
            cfg.model.roi_relation_head.use_gt_object_label = False

        elif task == "sgcls":
            cfg.model.roi_relation_head.use_gt_box = True
            cfg.model.roi_relation_head.use_gt_object_label = False

        elif task == "predcls":
            cfg.model.roi_relation_head.use_gt_box = True
            cfg.model.roi_relation_head.use_gt_object_label = True

        else:
            raise ValueError(f"Unknown task: {task}")

    assert_mode(cfg, task)


def build_config(
    *,
    config_file: str,
    cli_opts: Iterable[str],
    task: str,
    sampled_overrides: Mapping[str, Any],
    max_epochs: int | None,
    max_images: int | None,
    output_dir: str | None,
    apply_tuning_budget: bool,
) -> DictConfig:
    """Load the YAML afresh and apply all overrides in deterministic order."""
    cfg = load_config_from_file(config_file)

    if cli_opts:
        cfg = update_config_from_list(cfg, list(cli_opts))

    apply_overrides(cfg, FIXED_OVERRIDES)
    apply_overrides(cfg, sampled_overrides)
    set_task_mode(cfg, task)

    with open_dict(cfg):
        if output_dir is not None:
            cfg.output_dir = output_dir

        if apply_tuning_budget:
            if max_epochs is not None and max_epochs > 0:
                cfg.solver.max_epoch = int(max_epochs)

            # Keep approximately the same number of training images even when
            # solver.ims_per_batch is itself being tuned.
            if max_images is not None and max_images > 0:
                batch_size = int(cfg.solver.ims_per_batch)
                cfg.solver.max_iter = max(
                    1,
                    math.ceil(int(max_images) / batch_size),
                )

            # Cached predictions would make post-processing trials incomparable.
            cfg.test.allow_load_from_cache = False

    # Ray workers do not necessarily expose a real terminal stream.
    # This avoids sys.stdout.isatty() errors in SGG-Benchmark's trainer.
    with open_dict(cfg.solver):
        cfg.solver.detached_logging = True

    return cfg


def numeric_mean(value: Any) -> float:
    """Recursively average numeric values from metric dictionaries/lists."""
    numbers: list[float] = []

    def collect(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.numel() > 0:
                numbers.extend(
                    float(x)
                    for x in item.detach().cpu().reshape(-1).tolist()
                )
            return

        if isinstance(item, np.ndarray):
            if item.size > 0:
                numbers.extend(float(x) for x in item.reshape(-1).tolist())
            return

        if isinstance(item, Mapping):
            for child in item.values():
                collect(child)
            return

        if isinstance(item, (list, tuple, set)):
            for child in item:
                collect(child)
            return

        if isinstance(item, (int, float, np.integer, np.floating)):
            numbers.append(float(item))

    collect(value)

    if not numbers:
        raise ValueError(f"Metric contains no numeric values: {value!r}")

    result = float(np.mean(numbers))

    if not math.isfinite(result):
        raise ValueError(f"Metric is not finite: {result}")

    return result


def resolve_metric_key(
    *,
    cfg: DictConfig,
    mode: str,
    requested_metric: str | None,
    explicit_metric_key: str | None,
) -> str:
    """Resolve an alias such as mR into a validation-result dictionary key."""
    if explicit_metric_key:
        return explicit_metric_key

    metric = requested_metric or str(cfg.metric_to_track)

    if metric in METRIC_SUFFIXES:
        return mode + METRIC_SUFFIXES[metric]

    # Permit direct keys or suffixes for custom metrics.
    if metric.startswith(mode):
        return metric

    if metric.startswith("_"):
        return mode + metric

    supported = ", ".join(METRIC_SUFFIXES)

    raise ValueError(
        f"Unknown metric '{metric}'. Use one of [{supported}] or pass "
        "--metric-key with the exact key returned by validation."
    )


def better(score: float, best_score: float, objective_mode: str) -> bool:
    return score > best_score if objective_mode == "max" else score < best_score


def initial_best(objective_mode: str) -> float:
    return -math.inf if objective_mode == "max" else math.inf


# =============================================================================
# SGG TRAINING SETUP
# =============================================================================

def build_slow_heads(cfg: DictConfig) -> list[str]:
    slow_heads: list[str] = []
    predictor = str(cfg.model.roi_relation_head.predictor)

    if predictor == "IMPPredictor":
        slow_heads.extend(
            [
                "roi_heads.relation.box_feature_extractor",
                "roi_heads.relation.union_feature_extractor.feature_extractor",
            ]
        )

    elif predictor == "SquatPredictor":
        slow_heads.append(
            "roi.heads.relation.predictor.context_layer.mask_predictor"
        )

    if not bool(cfg.model.backbone.freeze):
        slow_heads.append("backbone")

    return slow_heads


def prepare_model_for_relation_training(model: torch.nn.Module) -> None:
    """
    Match the Hydra REACT++ training path:
    relation head in train mode, YOLO detector backbone in eval mode.
    """
    if hasattr(model, "roi_heads"):
        model.roi_heads.train()
    else:
        model.train()

    if hasattr(model, "backbone"):
        model.backbone.eval()

    relation = getattr(getattr(model, "roi_heads", None), "relation", None)

    if relation is not None:
        for parameter in relation.parameters():
            parameter.requires_grad = True


def create_checkpointer(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: str,
    logger: Any,
) -> DetectronCheckpointer:
    return DetectronCheckpointer(
        cfg,
        model,
        optimizer,
        scheduler,
        output_dir,
        True,
        custom_scheduler=True,
        logger=logger,
    )


def configure_trial(
    trial_config: Dict[str, Any],
    *,
    config_file: str,
    cli_opts: list[str],
    task: str,
    max_epochs: int,
    max_images: int,
    project_root: str,
):
    """Construct all SGG-Benchmark objects for one independent Ray trial."""
    os.chdir(project_root)

    context = tune.get_context()
    trial_dir = Path(context.get_trial_dir()).resolve()
    trial_id = context.get_trial_id()

    output_dir = trial_dir / "sgg"
    mkdir(str(output_dir))

    cfg = build_config(
        config_file=config_file,
        cli_opts=cli_opts,
        task=task,
        sampled_overrides=trial_config["overrides"],
        max_epochs=max_epochs,
        max_images=max_images,
        output_dir=str(output_dir),
        apply_tuning_budget=True,
    )

    set_seed(seed=int(cfg.seed))

    logger = setup_logger(
        f"sgg_tune_{trial_id}",
        str(output_dir),
        0,
        verbose=cfg.verbose,
        steps=True,
    )

    logger.info(
        "Trial overrides:\n%s",
        json.dumps(
            trial_config["overrides"],
            indent=2,
            default=str,
        ),
    )

    # Load the training dataset first so class counts can be inferred.
    train_loader = make_data_loader(
        cfg,
        mode="train",
        is_distributed=False,
    )

    if isinstance(train_loader, (list, tuple)):
        if len(train_loader) != 1:
            raise RuntimeError(
                "Expected one training loader, got "
                f"{len(train_loader)} loaders."
            )

        train_loader = train_loader[0]

    dataset = train_loader.dataset

    num_obj_classes = (
        len(dataset.ind_to_classes)
        if hasattr(dataset, "ind_to_classes")
        else int(cfg.model.roi_box_head.num_classes)
    )

    num_rel_classes = (
        len(dataset.ind_to_predicates)
        if hasattr(dataset, "ind_to_predicates")
        else int(cfg.model.roi_relation_head.num_classes)
    )

    with open_dict(cfg):
        cfg.model.roi_box_head.num_classes = num_obj_classes
        cfg.model.roi_relation_head.num_classes = num_rel_classes

    OmegaConf.save(cfg, output_dir / "config_trial.yaml")

    model = build_detection_model(cfg)
    device = torch.device(str(cfg.model.device))
    model.to(device)

    optimizer = make_optimizer(
        cfg,
        model,
        logger,
        slow_heads=build_slow_heads(cfg),
        slow_ratio=float(getattr(cfg.solver, "slow_ratio", 2.5)),
        rl_factor=1.0,
    )

    raw_iters = len(train_loader)
    max_iter = int(getattr(cfg.solver, "max_iter", 0))

    iters_per_epoch = (
        min(raw_iters, max_iter)
        if max_iter > 0
        else raw_iters
    )

    scheduler = make_lr_scheduler(
        cfg,
        optimizer,
        logger,
        iters_per_epoch=iters_per_epoch,
    )

    use_amp = str(cfg.dtype).lower() == "float16"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    val_loaders = make_data_loader(
        cfg,
        mode="val",
        is_distributed=False,
    )

    if not isinstance(val_loaders, (list, tuple)):
        val_loaders = [val_loaders]

    checkpointer = create_checkpointer(
        cfg,
        model,
        optimizer,
        scheduler,
        str(output_dir),
        logger,
    )

    pretrained = str(
        cfg.model.get("pretrained_detector_ckpt", "") or ""
    )

    if pretrained:
        pretrained_path = Path(pretrained)

        if not pretrained_path.is_absolute():
            pretrained_path = Path(project_root) / pretrained_path

        if not pretrained_path.exists():
            raise FileNotFoundError(
                "Pretrained detector checkpoint not found: "
                f"{pretrained_path}"
            )

        logger.info(
            f"Loading pretrained backbone detector from: {pretrained_path}"
        )

        checkpointer.load_backbone(str(pretrained_path))

    prepare_model_for_relation_training(model)

    return (
        cfg,
        model,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        val_loaders,
        checkpointer,
        logger,
        device,
        output_dir,
    )


# =============================================================================
# RAY TRIAL
# =============================================================================

def train_trial(
    trial_config: Dict[str, Any],
    *,
    config_file: str,
    cli_opts: list[str],
    task: str,
    requested_metric: str | None,
    explicit_metric_key: str | None,
    objective_mode: str,
    max_epochs: int,
    max_images: int,
    project_root: str,
    save_checkpoints: bool,
) -> None:
    """Ray trainable: one complete SGG training/validation experiment."""
    model = None
    output_dir = None

    try:
        (
            cfg,
            model,
            optimizer,
            scheduler,
            scaler,
            train_loader,
            val_loaders,
            checkpointer,
            logger,
            device,
            output_dir,
        ) = configure_trial(
            trial_config,
            config_file=config_file,
            cli_opts=cli_opts,
            task=task,
            max_epochs=max_epochs,
            max_images=max_images,
            project_root=project_root,
        )

        mode = get_mode(cfg)

        metric_key = resolve_metric_key(
            cfg=cfg,
            mode=mode,
            requested_metric=requested_metric,
            explicit_metric_key=explicit_metric_key,
        )

        best_score = initial_best(objective_mode)
        previous_checkpoint: Path | None = None

        max_epoch = int(cfg.solver.max_epoch)
        use_amp = str(cfg.dtype).lower() == "float16"

        for epoch in range(max_epoch):
            prepare_model_for_relation_training(model)

            train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                epoch=epoch,
                logger=logger,
                cfg=cfg,
                scaler=scaler,
                use_wandb=False,
                use_amp=use_amp,
                scheduler=scheduler,
            )

            val_result = run_val(
                cfg,
                model,
                val_loaders,
                distributed=False,
                logger=logger,
                device=device,
            )

            if metric_key not in val_result:
                available = ", ".join(
                    sorted(str(key) for key in val_result)
                )

                raise KeyError(
                    f"Validation metric '{metric_key}' was not returned. "
                    f"Available keys: {available}"
                )

            score = numeric_mean(val_result[metric_key])
            current_lr = float(optimizer.param_groups[0]["lr"])

            if better(score, best_score, objective_mode):
                best_score = score

                if save_checkpoints:
                    checkpoint_name = f"tune_best_epoch_{epoch}"

                    checkpointer.save(
                        checkpoint_name,
                        epoch=epoch,
                        best_metric=best_score,
                        metric_key=metric_key,
                    )

                    new_checkpoint = (
                        output_dir / f"{checkpoint_name}.pth"
                    )

                    if (
                        previous_checkpoint is not None
                        and previous_checkpoint != new_checkpoint
                        and previous_checkpoint.exists()
                    ):
                        previous_checkpoint.unlink()

                    previous_checkpoint = new_checkpoint

            # Iter-based schedulers are stepped inside train_one_epoch.
            if not getattr(scheduler, "_is_iter_based", False):
                if (
                    str(cfg.solver.schedule.type)
                    == "WarmupReduceLROnPlateau"
                ):
                    scheduler.step(score, epoch=epoch)
                else:
                    scheduler.step()

            train.report(
                {
                    "score": score,
                    "best_score": best_score,
                    "epoch": epoch + 1,
                    "metric_key": metric_key,
                    "lr": current_lr,
                    "oom": 0,
                }
            )

    except torch.cuda.OutOfMemoryError:
        # Treat OOM configurations as very poor observations instead of
        # terminating the complete tuning run.
        bad_score = (
            -1.0e30
            if objective_mode == "max"
            else 1.0e30
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train.report(
            {
                "score": bad_score,
                "best_score": bad_score,
                "epoch": 0,
                "metric_key": (
                    explicit_metric_key
                    or requested_metric
                    or ""
                ),
                "lr": float("nan"),
                "oom": 1,
            }
        )

    finally:
        if model is not None:
            del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# DRIVER
# =============================================================================

def validate_user_configuration(
    *,
    config_file: str,
    cli_opts: list[str],
    task: str,
) -> None:
    """Fail before starting Ray if an override path or distribution is invalid."""
    cfg = load_config_from_file(config_file)

    if cli_opts:
        cfg = update_config_from_list(cfg, cli_opts)

    set_task_mode(cfg, task)

    print("\nFixed overrides:")

    for path, value in FIXED_OVERRIDES.items():
        old = ensure_path_exists(cfg, path)

        print(
            f"  {normalize_path(path)}: "
            f"{old!r} -> {value!r}"
        )

    apply_overrides(cfg, FIXED_OVERRIDES)

    print("\nTunable parameters:")

    for path, spec in SEARCH_SPACE.items():
        old = ensure_path_exists(cfg, path)

        # Validate the distribution specification.
        make_domain(spec)

        print(
            f"  {normalize_path(path)}: current={old!r}, "
            f"distribution={dict(spec)!r}"
        )

    if not SEARCH_SPACE:
        raise ValueError("SEARCH_SPACE is empty.")


def save_best_artifacts(
    *,
    best_result: Any,
    experiment_dir: Path,
    config_file: str,
    cli_opts: list[str],
    task: str,
    requested_metric: str | None,
    explicit_metric_key: str | None,
    objective_mode: str,
) -> None:
    """Save the best overrides and a full-training YAML based on the winner."""
    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_overrides = dict(
        best_result.config["overrides"]
    )

    # Preserve the original YAML training duration.
    best_cfg = build_config(
        config_file=config_file,
        cli_opts=cli_opts,
        task=task,
        sampled_overrides=best_overrides,
        max_epochs=None,
        max_images=None,
        output_dir=None,
        apply_tuning_budget=False,
    )

    OmegaConf.save(
        best_cfg,
        experiment_dir / "best_config_full_training.yaml",
    )

    override_document = {
        "fixed_overrides": FIXED_OVERRIDES,
        "tuned_overrides": best_overrides,
        "objective": {
            "metric": requested_metric,
            "metric_key": explicit_metric_key,
            "mode": objective_mode,
            "score": best_result.metrics.get("score"),
        },
    }

    OmegaConf.save(
        OmegaConf.create(override_document),
        experiment_dir / "best_overrides.yaml",
    )

    summary = {
        "score": best_result.metrics.get("score"),
        "best_score": best_result.metrics.get("best_score"),
        "metric_key": best_result.metrics.get("metric_key"),
        "epoch": best_result.metrics.get("epoch"),
        "oom": best_result.metrics.get("oom", 0),
        "trial_path": str(best_result.path),
        "overrides": best_overrides,
    }

    with (
        experiment_dir / "best_result.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    try:
        best_result_df = best_result.metrics_dataframe

        if best_result_df is not None:
            best_result_df.to_csv(
                experiment_dir / "best_trial_history.csv",
                index=False,
            )

    except Exception:
        # The exact Result API varies slightly across Ray versions.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic Ray Tune + Optuna tuning for Hydra YAML configs in "
            "SGG-Benchmark."
        )
    )

    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the base Hydra YAML, for example REACT++.yaml.",
    )

    parser.add_argument(
        "--task",
        choices=["sgdet", "sgcls", "predcls"],
        default="sgdet",
    )

    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Metric alias to optimize: mR, R, F1, zR, ng-zR, ng-R or "
            "ng-mR. Defaults to metric_to_track in the YAML."
        ),
    )

    parser.add_argument(
        "--metric-key",
        default=None,
        help=(
            "Exact key returned by run_val, for custom objectives. "
            "Overrides --metric."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["max", "min"],
        default="max",
        help="Whether the objective score must be maximized or minimized.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of configurations requested from Optuna.",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Maximum epochs per tuning trial.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
        help=(
            "Approximate maximum training images per epoch. "
            "Use 0 to process the complete training loader."
        ),
    )

    parser.add_argument(
        "--grace-period",
        type=int,
        default=1,
        help="Minimum reported epochs before ASHA may stop a trial.",
    )

    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=3,
        help="ASHA reduction factor.",
    )

    parser.add_argument(
        "--cpus-per-trial",
        type=float,
        default=6,
    )

    parser.add_argument(
        "--gpus-per-trial",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=None,
        help="Optional explicit concurrency limit.",
    )

    parser.add_argument(
        "--storage-path",
        default="./ray_results",
        help="Local directory used by Ray Tune.",
    )

    parser.add_argument(
        "--ray-temp-dir",
        default=None,
        help=(
            "Directory for Ray session files and object spilling. "
            "Defaults to <storage-path>/_ray_tmp. Put it on a filesystem "
            "with sufficient free space."
        ),
    )

    parser.add_argument(
        "--experiment-name",
        default="sgg_hydra_tuning",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used by Optuna's TPE sampler.",
    )

    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help=(
            "Save the best model inside every trial. Disabled by default "
            "to avoid large disk usage."
        ),
    )

    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Optional fixed YACS-style key/value pairs after --opts, e.g. "
            "--opts DTYPE float16 SOLVER.WEIGHT_DECAY 0.01"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_epochs < 1:
        raise ValueError(
            "--max-epochs must be at least 1."
        )

    if args.num_samples < 1:
        raise ValueError(
            "--num-samples must be at least 1."
        )

    if (
        args.grace_period < 1
        or args.grace_period > args.max_epochs
    ):
        raise ValueError(
            "--grace-period must be between 1 and --max-epochs."
        )

    if args.reduction_factor <= 1:
        raise ValueError(
            "--reduction-factor must be greater than 1."
        )

    config_file = str(
        Path(args.config_file).expanduser().resolve()
    )

    if not Path(config_file).is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_file}"
        )

    storage_path = (
        Path(args.storage_path)
        .expanduser()
        .resolve()
    )

    storage_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_dir = (
        storage_path / args.experiment_name
    )

    ray_temp_dir = (
        Path(args.ray_temp_dir).expanduser().resolve()
        if args.ray_temp_dir
        else storage_path / "_ray_tmp"
    )

    ray_spill_dir = (
        ray_temp_dir / "object_spilling"
    )

    ray_temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ray_spill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ray.is_initialized():
        ray.init(
            _temp_dir=str(ray_temp_dir),
            object_spilling_directory=str(ray_spill_dir),
        )

    try:
        validate_user_configuration(
            config_file=config_file,
            cli_opts=args.opts,
            task=args.task,
        )

        param_space = {
            "overrides": {
                normalize_path(path): make_domain(spec)
                for path, spec in SEARCH_SPACE.items()
            }
        }

        sampler = TPESampler(
            seed=args.seed
        )

        # metric and mode are intentionally omitted here.
        # They are defined once in TuneConfig.
        search_algorithm = OptunaSearch(
            sampler=sampler,
        )

        # metric and mode are intentionally omitted here.
        # They are defined once in TuneConfig.
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=args.max_epochs,
            grace_period=args.grace_period,
            reduction_factor=args.reduction_factor,
        )

        tune_config_kwargs: Dict[str, Any] = {
            "metric": "score",
            "mode": args.mode,
            "search_alg": search_algorithm,
            "scheduler": scheduler,
            "num_samples": args.num_samples,
        }

        if args.max_concurrent_trials is not None:
            tune_config_kwargs["max_concurrent_trials"] = (
                args.max_concurrent_trials
            )

        trainable = tune.with_parameters(
            train_trial,
            config_file=config_file,
            cli_opts=args.opts,
            task=args.task,
            requested_metric=args.metric,
            explicit_metric_key=args.metric_key,
            objective_mode=args.mode,
            max_epochs=args.max_epochs,
            max_images=args.max_images,
            project_root=str(PROJECT_ROOT),
            save_checkpoints=args.save_checkpoints,
        )

        trainable = tune.with_resources(
            trainable,
            resources={
                "cpu": args.cpus_per_trial,
                "gpu": args.gpus_per_trial,
            },
        )

        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                **tune_config_kwargs
            ),
            run_config=RunConfig(
                name=args.experiment_name,
                storage_path=str(storage_path),

                # Keep stdout as a normal stream. Ray's Tee wrapper caused
                # SGG-Benchmark's sys.stdout.isatty() call to fail.
                log_to_file=False,

                verbose=1,

                callbacks=[
                    JsonLoggerCallback(),
                    CSVLoggerCallback(),
                    TBXLoggerCallback(),
                ],
            ),
        )

        results = tuner.fit()

        best_result = results.get_best_result(
            metric="score",
            mode=args.mode,
            filter_nan_and_inf=True,
        )

        save_best_artifacts(
            best_result=best_result,
            experiment_dir=experiment_dir,
            config_file=config_file,
            cli_opts=args.opts,
            task=args.task,
            requested_metric=args.metric,
            explicit_metric_key=args.metric_key,
            objective_mode=args.mode,
        )

        try:
            results.get_dataframe().to_csv(
                experiment_dir / "all_trials.csv",
                index=False,
            )

        except Exception:
            pass

        print("\nBest trial")
        print(
            f"  score: "
            f"{best_result.metrics.get('score')}"
        )
        print(
            f"  metric key: "
            f"{best_result.metrics.get('metric_key')}"
        )
        print(
            f"  trial path: "
            f"{best_result.path}"
        )
        print("  overrides:")

        for path, value in (
            best_result.config["overrides"].items()
        ):
            print(f"    {path}: {value}")

        print(
            f"\nSaved summary in: {experiment_dir}"
        )

    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
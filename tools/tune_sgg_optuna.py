#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for Hydra/OmegaConf SGG-Benchmark configs.

Place this file in:
    SGG-Benchmark/tools/optimize_optuna.py

Main differences from a Ray Tune based tuner:

* Optuna is the tuning engine, not only the sampler.
* Results are persisted in an SQLite database that can be opened live with
  optuna-dashboard.
* The objective returned to Optuna is produced by a fresh post-training
  evaluation: the selected checkpoint is saved, the training model is freed,
  a new model is built, the checkpoint is reloaded, and inference() is run on
  the requested validation/test split, following relation_eval_hydra.py.
* Per-epoch validation is still used internally to select the checkpoint and
  to drive validation-dependent schedulers, but it is not the final Optuna
  objective.

The original YAML is never modified. Edit SEARCH_SPACE and FIXED_OVERRIDES to
add or remove parameters.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

# Make the repository importable when this script is launched from tools/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SGG-Benchmark asks for this import before most other project modules.
from sgg_benchmark.utils.env import setup_environment  # noqa: F401,E402

import numpy as np
import optuna
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from sgg_benchmark.config import (  # noqa: E402
    load_config_from_file,
    update_config_from_list,
)
from sgg_benchmark.data import (  # noqa: E402
    get_dataset_statistics,
    make_data_loader,
)
from sgg_benchmark.engine.inference import inference  # noqa: E402
from sgg_benchmark.engine.trainer import (  # noqa: E402
    assert_mode,
    get_mode,
    run_val,
    train_one_epoch,
)
from sgg_benchmark.modeling.detector import build_detection_model  # noqa: E402
from sgg_benchmark.solver import make_lr_scheduler, make_optimizer  # noqa: E402
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer  # noqa: E402
from sgg_benchmark.utils.comm import synchronize  # noqa: E402
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
#   Both bounds are inclusive in this Optuna script.
#
# quniform:
#   {"type": "quniform", "low": 0.1, "high": 0.9, "q": 0.1}
#
# qrandint:
#   {"type": "qrandint", "low": 32, "high": 1024, "q": 32}
#
# Paths are OmegaConf dot paths and are case-insensitive in this script.
# Every path is checked before Optuna starts.
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

    # Detector/post-processing parameters
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

# Values always applied before sampled overrides.
FIXED_OVERRIDES: Dict[str, Any] = {
    "solver.optimizer": "ADAMW",

    # Examples:
    # "dtype": "float16",
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

_MISSING = object()


# =============================================================================
# GENERIC UTILITIES
# =============================================================================

def normalize_path(path: str) -> str:
    """Convert a YACS-style or Hydra-style path to lowercase Hydra notation."""
    return path.strip().lower()


def json_safe(value: Any) -> Any:
    """Convert nested tensors/arrays/config objects to JSON-safe values."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(child)
            for key, child in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(child) for child in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def numeric_values(value: Any) -> list[float]:
    """Recursively collect finite numeric values."""
    numbers: list[float] = []

    def collect(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.numel() > 0:
                for child in item.detach().cpu().reshape(-1).tolist():
                    collect(child)
            return

        if isinstance(item, np.ndarray):
            if item.size > 0:
                for child in item.reshape(-1).tolist():
                    collect(child)
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
            number = float(item)
            if math.isfinite(number):
                numbers.append(number)

    collect(value)
    return numbers


def numeric_mean(value: Any) -> float:
    numbers = numeric_values(value)

    if not numbers:
        raise ValueError(
            f"Metric contains no finite numeric values: {value!r}"
        )

    return float(np.mean(numbers))


def better(
    score: float,
    best_score: float,
    direction: str,
) -> bool:
    if direction == "maximize":
        return score > best_score

    return score < best_score


def initial_best(direction: str) -> float:
    if direction == "maximize":
        return -math.inf

    return math.inf


def remove_file_quietly(path: Path | None) -> None:
    if path is None:
        return

    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def clear_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# CONFIG AND SEARCH SPACE
# =============================================================================

def ensure_path_exists(
    cfg: DictConfig,
    path: str,
) -> Any:
    normalized = normalize_path(path)

    value = OmegaConf.select(
        cfg,
        normalized,
        default=_MISSING,
    )

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
    for raw_path, value in overrides.items():
        path = normalize_path(raw_path)

        if validate:
            ensure_path_exists(cfg, path)

        OmegaConf.update(
            cfg,
            path,
            value,
            merge=False,
        )

    return cfg


def set_task_mode(
    cfg: DictConfig,
    task: str,
) -> None:
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


def suggest_parameter(
    trial: optuna.Trial,
    path: str,
    spec: Mapping[str, Any],
) -> Any:
    """Sample one SEARCH_SPACE entry through Optuna."""
    name = normalize_path(path)
    kind = str(spec["type"]).lower()

    if kind == "choice":
        values = list(spec["values"])

        if not values:
            raise ValueError(
                f"{path}: choice requires non-empty values"
            )

        return trial.suggest_categorical(
            name,
            values,
        )

    if kind == "uniform":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
        )

    if kind == "loguniform":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=True,
        )

    if kind == "randint":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
        )

    if kind == "quniform":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=float(spec["q"]),
        )

    if kind == "qloguniform":
        raise ValueError(
            f"{path}: Optuna does not allow log=True together with step. "
            "Use loguniform or a categorical grid instead."
        )

    if kind == "qrandint":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec["q"]),
        )

    raise ValueError(
        f"Unsupported distribution type '{kind}' for {path}. "
        "Use choice, uniform, loguniform, randint, quniform or qrandint."
    )


def validate_distribution_spec(
    path: str,
    spec: Mapping[str, Any],
) -> None:
    kind = str(spec.get("type", "")).lower()

    supported = {
        "choice",
        "uniform",
        "loguniform",
        "randint",
        "quniform",
        "qrandint",
    }

    if kind not in supported:
        raise ValueError(
            f"{path}: unsupported distribution '{kind}'. "
            f"Supported: {sorted(supported)}"
        )

    if kind == "choice":
        if not list(spec.get("values", [])):
            raise ValueError(
                f"{path}: choice requires non-empty values"
            )

        return

    if "low" not in spec or "high" not in spec:
        raise ValueError(
            f"{path}: {kind} requires low and high"
        )

    if float(spec["low"]) > float(spec["high"]):
        raise ValueError(
            f"{path}: low must be <= high"
        )

    if (
        kind in {"quniform", "qrandint"}
        and float(spec.get("q", 0)) <= 0
    ):
        raise ValueError(
            f"{path}: {kind} requires q > 0"
        )


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
    """Load the YAML afresh and apply all overrides deterministically."""
    cfg = load_config_from_file(config_file)

    if cli_opts:
        cfg = update_config_from_list(
            cfg,
            list(cli_opts),
        )

    apply_overrides(
        cfg,
        FIXED_OVERRIDES,
    )

    apply_overrides(
        cfg,
        sampled_overrides,
    )

    set_task_mode(
        cfg,
        task,
    )

    with open_dict(cfg):
        if output_dir is not None:
            cfg.output_dir = output_dir

        if apply_tuning_budget:
            if max_epochs is not None and max_epochs > 0:
                cfg.solver.max_epoch = int(max_epochs)

            if max_images is not None and max_images > 0:
                batch_size = max(
                    1,
                    int(cfg.solver.ims_per_batch),
                )

                cfg.solver.max_iter = max(
                    1,
                    math.ceil(
                        int(max_images) / batch_size
                    ),
                )

        # Cached predictions make trials incomparable when inference parameters
        # such as NMS or max detections are tuned.
        cfg.test.allow_load_from_cache = False

        # No Ray stdout wrapper is used in the Optuna-only runner.
        # Keep the structured SolverConfig unchanged: normal Python streams
        # already provide isatty(), so detached_logging is unnecessary.

    return cfg


def validate_user_configuration(
    *,
    config_file: str,
    cli_opts: list[str],
    task: str,
) -> None:
    cfg = load_config_from_file(config_file)

    if cli_opts:
        cfg = update_config_from_list(
            cfg,
            cli_opts,
        )

    set_task_mode(
        cfg,
        task,
    )

    print("\nFixed overrides:")

    for path, value in FIXED_OVERRIDES.items():
        old = ensure_path_exists(
            cfg,
            path,
        )

        print(
            f"  {normalize_path(path)}: "
            f"{old!r} -> {value!r}"
        )

    apply_overrides(
        cfg,
        FIXED_OVERRIDES,
    )

    print("\nTunable parameters:")

    for path, spec in SEARCH_SPACE.items():
        old = ensure_path_exists(
            cfg,
            path,
        )

        validate_distribution_spec(
            path,
            spec,
        )

        print(
            f"  {normalize_path(path)}: current={old!r}, "
            f"distribution={dict(spec)!r}"
        )

    if not SEARCH_SPACE:
        raise ValueError(
            "SEARCH_SPACE is empty."
        )


# =============================================================================
# METRIC UTILITIES
# =============================================================================

def resolve_metric_key(
    *,
    cfg: DictConfig,
    task_mode: str,
    requested_metric: str | None,
    explicit_metric_key: str | None,
) -> str:
    if explicit_metric_key:
        return explicit_metric_key

    metric = requested_metric or str(cfg.metric_to_track)

    if metric in METRIC_SUFFIXES:
        return task_mode + METRIC_SUFFIXES[metric]

    if metric.startswith(task_mode):
        return metric

    if metric.startswith("_"):
        return task_mode + metric

    supported = ", ".join(METRIC_SUFFIXES)

    raise ValueError(
        f"Unknown metric '{metric}'. Use one of [{supported}] or pass "
        "--metric-key with the exact inference result key."
    )


def select_metric_at_k(
    value: Any,
    metric_k: int | None,
) -> float:
    """Select @K from a metric dict, or average all numeric values."""
    if metric_k is None or metric_k <= 0:
        return numeric_mean(value)

    if isinstance(value, Mapping):
        candidates: Sequence[Any] = (
            metric_k,
            str(metric_k),
            f"@{metric_k}",
            f"R@{metric_k}",
            f"mR@{metric_k}",
            f"F1@{metric_k}",
        )

        for key in candidates:
            if key in value:
                return numeric_mean(value[key])

        normalized: dict[str, Any] = {
            str(key).replace(" ", "").lower(): child
            for key, child in value.items()
        }

        suffix = str(metric_k)

        for key, child in normalized.items():
            if key.endswith(suffix):
                return numeric_mean(child)

        raise KeyError(
            f"Could not find metric cutoff @{metric_k}. "
            f"Available keys: {list(value.keys())}"
        )

    # Scalar metrics do not have a K dimension.
    return numeric_mean(value)


def find_metric_values(
    result: Any,
    metric_key: str,
) -> list[Any]:
    """
    Find metric_key recursively.

    Supports both direct output dictionaries and dictionaries nested by
    dataset name.
    """
    matches: list[Any] = []

    def visit(item: Any) -> None:
        if not isinstance(item, Mapping):
            return

        if metric_key in item:
            matches.append(item[metric_key])

        for child in item.values():
            if isinstance(child, Mapping):
                visit(child)

    visit(result)
    return matches


def objective_from_results(
    result: Any,
    *,
    metric_key: str,
    metric_k: int | None,
) -> float:
    matches = find_metric_values(
        result,
        metric_key,
    )

    if not matches:
        available: set[str] = set()

        def collect_keys(item: Any) -> None:
            if isinstance(item, Mapping):
                available.update(
                    str(key)
                    for key in item.keys()
                )

                for child in item.values():
                    collect_keys(child)

        collect_keys(result)

        raise KeyError(
            f"Evaluation metric '{metric_key}' was not returned. "
            f"Available keys include: {sorted(available)}"
        )

    scores = [
        select_metric_at_k(
            value,
            metric_k,
        )
        for value in matches
    ]

    score = float(np.mean(scores))

    if not math.isfinite(score):
        raise ValueError(
            f"Final evaluation score is not finite: {score}"
        )

    return score


# =============================================================================
# MODEL/TRAINING SETUP
# =============================================================================

def build_slow_heads(
    cfg: DictConfig,
) -> list[str]:
    slow_heads: list[str] = []
    predictor = str(
        cfg.model.roi_relation_head.predictor
    )

    if predictor == "IMPPredictor":
        slow_heads.extend(
            [
                "roi_heads.relation.box_feature_extractor",
                (
                    "roi_heads.relation."
                    "union_feature_extractor.feature_extractor"
                ),
            ]
        )

    elif predictor == "SquatPredictor":
        slow_heads.append(
            "roi.heads.relation.predictor.context_layer.mask_predictor"
        )

    if not bool(cfg.model.backbone.freeze):
        slow_heads.append("backbone")

    return slow_heads


def prepare_model_for_relation_training(
    model: torch.nn.Module,
) -> None:
    if hasattr(model, "roi_heads"):
        model.roi_heads.train()
    else:
        model.train()

    # YOLO backbones should remain in eval mode so their head returns decoded
    # predictions suitable for NMS while the relation head is trained.
    if hasattr(model, "backbone"):
        model.backbone.eval()

    relation = getattr(
        getattr(model, "roi_heads", None),
        "relation",
        None,
    )

    if relation is not None:
        for parameter in relation.parameters():
            parameter.requires_grad = True


def enable_inplace_relu(
    module: torch.nn.Module,
) -> None:
    """Match relation_eval_hydra.py's memory optimization."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(
                module,
                name,
                torch.nn.ReLU(inplace=True),
            )
        else:
            enable_inplace_relu(child)


def update_class_counts_from_loader(
    cfg: DictConfig,
    loader: Any,
) -> None:
    dataset = loader.dataset

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
        cfg.model.roi_box_head.num_classes = int(
            num_obj_classes
        )

        cfg.model.roi_relation_head.num_classes = int(
            num_rel_classes
        )


def as_loader_list(
    loaders: Any,
) -> list[Any]:
    if isinstance(loaders, (list, tuple)):
        return list(loaders)

    return [loaders]


def create_checkpointer(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    output_dir: str,
    logger: Any,
) -> DetectronCheckpointer:
    if optimizer is None or scheduler is None:
        return DetectronCheckpointer(
            cfg,
            model,
            save_dir=output_dir,
            logger=logger,
        )

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


def maybe_load_pretrained_backbone(
    cfg: DictConfig,
    checkpointer: DetectronCheckpointer,
    logger: Any,
    project_root: Path,
) -> None:
    pretrained = str(
        cfg.model.get(
            "pretrained_detector_ckpt",
            "",
        )
        or ""
    )

    if not pretrained:
        return

    path = Path(pretrained).expanduser()

    if not path.is_absolute():
        path = project_root / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Pretrained detector checkpoint not found: {path}"
        )

    logger.info(
        f"Loading pretrained backbone detector from: {path}"
    )

    checkpointer.load_backbone(
        str(path)
    )


# =============================================================================
# INDEPENDENT POST-TRAINING EVALUATION
# =============================================================================

def dataset_names_for_split(
    cfg: DictConfig,
    split: str,
    count: int,
) -> list[str]:
    datasets_cfg = cfg.get(
        "datasets",
        {},
    )

    names = list(
        datasets_cfg.get(
            split,
            [],
        )
        or []
    )

    if not names:
        base_name = str(
            datasets_cfg.get(
                "name",
                "",
            )
            or ""
        )

        if base_name:
            names = [
                f"{base_name}_{split}"
            ]

    if not names:
        names = [
            f"{split}_{index}"
            for index in range(count)
        ]

    if len(names) < count:
        names.extend(
            f"{split}_{index}"
            for index in range(
                len(names),
                count,
            )
        )

    return names[:count]


def evaluate_checkpoint_fresh(
    *,
    config_file: str,
    cli_opts: list[str],
    task: str,
    sampled_overrides: Mapping[str, Any],
    checkpoint_path: Path,
    evaluation_split: str,
    evaluation_output_dir: Path,
    metric_key: str,
    metric_k: int | None,
    logger: Any,
    save_predictions: bool,
) -> tuple[float, Dict[str, Any]]:
    """
    Rebuild and evaluate a checkpoint using the relation_eval_hydra.py pattern.

    No training-model object is reused. This makes the Optuna objective reflect
    the serialised checkpoint and the normal inference pipeline.
    """
    eval_cfg = build_config(
        config_file=config_file,
        cli_opts=cli_opts,
        task=task,
        sampled_overrides=sampled_overrides,
        max_epochs=None,
        max_images=None,
        output_dir=str(evaluation_output_dir),
        apply_tuning_budget=False,
    )

    evaluation_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    OmegaConf.save(
        eval_cfg,
        evaluation_output_dir / "hydra_config.yaml",
    )

    loaders = as_loader_list(
        make_data_loader(
            eval_cfg,
            mode=evaluation_split,
            is_distributed=False,
        )
    )

    if not loaders:
        raise RuntimeError(
            f"No data loader returned for split '{evaluation_split}'."
        )

    update_class_counts_from_loader(
        eval_cfg,
        loaders[0],
    )

    model = build_detection_model(
        eval_cfg
    )

    device = torch.device(
        str(eval_cfg.model.device)
    )

    model.to(device)

    enable_inplace_relu(
        model
    )

    checkpointer = create_checkpointer(
        eval_cfg,
        model,
        optimizer=None,
        scheduler=None,
        output_dir=str(evaluation_output_dir),
        logger=logger,
    )

    logger.info(
        f"Post-training evaluation checkpoint: {checkpoint_path}"
    )

    checkpointer.load(
        str(checkpoint_path)
    )

    backbone_type = str(
        eval_cfg.model.backbone.get(
            "type",
            "",
        )
    ).lower()

    if (
        "world" in backbone_type
        and hasattr(model.backbone, "load_txt_feats")
    ):
        logger.info(
            "Loading text embeddings for YOLO World evaluation"
        )

        stats = get_dataset_statistics(
            eval_cfg
        )

        object_classes = stats["obj_classes"][1:]

        model.backbone.load_txt_feats(
            object_classes
        )

    model.eval()

    if hasattr(model, "backbone"):
        model.backbone.eval()

    if hasattr(model, "roi_heads"):
        model.roi_heads.eval()

    iou_types: tuple[str, ...] = ("bbox",)

    if bool(eval_cfg.model.relation_on):
        iou_types += ("relations",)

    if bool(
        eval_cfg.model.get(
            "attribute_on",
            False,
        )
    ):
        iou_types += ("attributes",)

    dataset_names = dataset_names_for_split(
        eval_cfg,
        evaluation_split,
        len(loaders),
    )

    use_amp = (
        str(eval_cfg.dtype).lower()
        == "float16"
    )

    all_results: Dict[str, Any] = {}

    try:
        for dataset_name, loader in zip(
            dataset_names,
            loaders,
        ):
            dataset_output = (
                evaluation_output_dir
                / dataset_name
            )

            output_folder = (
                str(dataset_output)
                if save_predictions
                else None
            )

            if output_folder is not None:
                mkdir(output_folder)

            logger.info(
                f"Independent evaluation on {dataset_name} "
                f"(split={evaluation_split})"
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(
                    use_amp
                    and device.type == "cuda"
                ),
            ):
                result = inference(
                    eval_cfg,
                    model,
                    loader,
                    dataset_name=dataset_name,
                    iou_types=iou_types,
                    box_only=bool(
                        eval_cfg.model.get(
                            "rpn_only",
                            False,
                        )
                    ),
                    device=str(
                        eval_cfg.model.device
                    ),
                    expected_results=eval_cfg.test.get(
                        "expected_results",
                        [],
                    ),
                    expected_results_sigma_tol=eval_cfg.test.get(
                        "expected_results_sigma_tol",
                        4,
                    ),
                    output_folder=output_folder,
                    logger=logger,
                    informative=bool(
                        eval_cfg.test.get(
                            "informative",
                            False,
                        )
                    ),
                )

            synchronize()

            all_results[dataset_name] = result

        score = objective_from_results(
            all_results,
            metric_key=metric_key,
            metric_k=metric_k,
        )

        return score, all_results

    finally:
        del model
        clear_cuda()


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def objective(
    trial: optuna.Trial,
    *,
    config_file: str,
    cli_opts: list[str],
    task: str,
    requested_metric: str | None,
    explicit_metric_key: str | None,
    metric_k: int | None,
    direction: str,
    max_epochs: int,
    max_images: int,
    experiment_dir: Path,
    evaluation_split: str,
    internal_validation_split: str,
    save_predictions: bool,
    keep_nonbest_checkpoints: bool,
) -> float:
    sampled_overrides = {
        normalize_path(path): suggest_parameter(
            trial,
            path,
            spec,
        )
        for path, spec in SEARCH_SPACE.items()
    }

    trial_dir = (
        experiment_dir
        / "trials"
        / f"trial_{trial.number:05d}"
    )

    train_dir = (
        trial_dir
        / "train"
    )

    eval_dir = (
        trial_dir
        / f"evaluation_{evaluation_split}"
    )

    train_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logger(
        f"sgg_optuna_trial_{trial.number}",
        str(trial_dir),
        0,
        filename="trial.log",
        verbose="INFO",
        steps=True,
    )

    trial.set_user_attr(
        "trial_dir",
        str(trial_dir),
    )

    trial.set_user_attr(
        "evaluation_split",
        evaluation_split,
    )

    trial.set_user_attr(
        "internal_validation_split",
        internal_validation_split,
    )

    trial.set_user_attr(
        "metric_k",
        metric_k if metric_k else "mean",
    )

    trial.set_user_attr(
        "oom",
        False,
    )

    with (
        trial_dir / "sampled_overrides.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            sampled_overrides,
            file,
            indent=2,
            default=str,
        )

    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any = None
    scaler: Any = None
    train_loader: Any = None
    internal_eval_loaders: list[Any] = []
    checkpointer: DetectronCheckpointer | None = None

    best_checkpoint: Path | None = None
    best_internal_score = initial_best(direction)
    metric_key = ""

    try:
        cfg = build_config(
            config_file=config_file,
            cli_opts=cli_opts,
            task=task,
            sampled_overrides=sampled_overrides,
            max_epochs=max_epochs,
            max_images=max_images,
            output_dir=str(train_dir),
            apply_tuning_budget=True,
        )

        set_seed(
            seed=(
                int(cfg.seed)
                + int(trial.number)
            )
        )

        train_loader = make_data_loader(
            cfg,
            mode="train",
            is_distributed=False,
        )

        if isinstance(
            train_loader,
            (list, tuple),
        ):
            if len(train_loader) != 1:
                raise RuntimeError(
                    "Expected one training loader, got "
                    f"{len(train_loader)}"
                )

            train_loader = train_loader[0]

        update_class_counts_from_loader(
            cfg,
            train_loader,
        )

        OmegaConf.save(
            cfg,
            train_dir / "hydra_config.yaml",
        )

        OmegaConf.save(
            cfg,
            train_dir / "config.yaml",
        )

        model = build_detection_model(
            cfg
        )

        device = torch.device(
            str(cfg.model.device)
        )

        model.to(device)

        optimizer = make_optimizer(
            cfg,
            model,
            logger,
            slow_heads=build_slow_heads(cfg),
            slow_ratio=float(
                getattr(
                    cfg.solver,
                    "slow_ratio",
                    2.5,
                )
            ),
            rl_factor=1.0,
        )

        raw_iters = len(train_loader)
        max_iter = int(
            getattr(
                cfg.solver,
                "max_iter",
                0,
            )
        )

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

        use_amp = (
            str(cfg.dtype).lower()
            == "float16"
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )

        internal_eval_loaders = as_loader_list(
            make_data_loader(
                cfg,
                mode=internal_validation_split,
                is_distributed=False,
            )
        )

        if not internal_eval_loaders:
            raise RuntimeError(
                "No data loader returned for internal validation split "
                f"'{internal_validation_split}'."
            )

        checkpointer = create_checkpointer(
            cfg,
            model,
            optimizer,
            scheduler,
            str(train_dir),
            logger,
        )

        maybe_load_pretrained_backbone(
            cfg,
            checkpointer,
            logger,
            PROJECT_ROOT,
        )

        task_mode = get_mode(cfg)

        metric_key = resolve_metric_key(
            cfg=cfg,
            task_mode=task_mode,
            requested_metric=requested_metric,
            explicit_metric_key=explicit_metric_key,
        )

        trial.set_user_attr(
            "metric_key",
            metric_key,
        )

        max_epoch = int(
            cfg.solver.max_epoch
        )

        internal_history: list[
            dict[str, Any]
        ] = []

        for epoch in range(max_epoch):
            logger.info(
                f"Optuna trial {trial.number}: "
                f"training epoch {epoch + 1}/{max_epoch}"
            )

            prepare_model_for_relation_training(
                model
            )

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

            # Internal evaluation is used only to choose the checkpoint and
            # to step schedulers that require an evaluation metric. The value
            # returned to Optuna is calculated later by a fresh evaluator.
            val_result = run_val(
                cfg,
                model,
                internal_eval_loaders,
                distributed=False,
                logger=logger,
                device=device,
            )

            internal_score = objective_from_results(
                val_result,
                metric_key=metric_key,
                metric_k=metric_k,
            )

            current_lr = float(
                optimizer.param_groups[0]["lr"]
            )

            internal_history.append(
                {
                    "epoch": epoch + 1,
                    "internal_validation_score": internal_score,
                    "lr": current_lr,
                }
            )

            # Intermediate values are visible in Optuna Dashboard. They are
            # not the final study objective.
            trial.report(
                internal_score,
                step=epoch + 1,
            )

            if better(
                internal_score,
                best_internal_score,
                direction,
            ):
                previous_checkpoint = best_checkpoint
                best_internal_score = internal_score

                checkpoint_name = (
                    f"best_model_epoch_{epoch}"
                )

                checkpointer.save(
                    checkpoint_name,
                    epoch=epoch,
                    best_metric=best_internal_score,
                    metric_key=metric_key,
                )

                best_checkpoint = (
                    train_dir
                    / f"{checkpoint_name}.pth"
                )

                if not keep_nonbest_checkpoints:
                    remove_file_quietly(
                        previous_checkpoint
                    )

            if not getattr(
                scheduler,
                "_is_iter_based",
                False,
            ):
                if (
                    str(cfg.solver.schedule.type)
                    == "WarmupReduceLROnPlateau"
                ):
                    scheduler.step(
                        internal_score,
                        epoch=epoch,
                    )
                else:
                    scheduler.step()

            if trial.should_prune():
                trial.set_user_attr(
                    "best_internal_validation_score",
                    best_internal_score,
                )

                raise optuna.TrialPruned(
                    f"Pruned after epoch {epoch + 1}; "
                    "internal validation score="
                    f"{internal_score:.6f}"
                )

        if (
            best_checkpoint is None
            or not best_checkpoint.exists()
        ):
            checkpoint_name = "model_final_tune"

            checkpointer.save(
                checkpoint_name,
                epoch=max_epoch - 1,
                best_metric=best_internal_score,
                metric_key=metric_key,
            )

            best_checkpoint = (
                train_dir
                / f"{checkpoint_name}.pth"
            )

        with (
            trial_dir
            / "internal_validation_history.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                internal_history,
                file,
                indent=2,
            )

        trial.set_user_attr(
            "best_internal_validation_score",
            float(best_internal_score),
        )

        trial.set_user_attr(
            "checkpoint",
            str(best_checkpoint),
        )

        # Release all training objects before rebuilding the model for the
        # independent relation_eval_hydra-style evaluation.
        del model
        model = None

        del optimizer
        del scheduler
        del scaler
        del train_loader
        del internal_eval_loaders
        del checkpointer

        optimizer = None
        scheduler = None
        scaler = None
        train_loader = None
        internal_eval_loaders = []
        checkpointer = None

        clear_cuda()

        final_score, final_results = evaluate_checkpoint_fresh(
            config_file=config_file,
            cli_opts=cli_opts,
            task=task,
            sampled_overrides=sampled_overrides,
            checkpoint_path=best_checkpoint,
            evaluation_split=evaluation_split,
            evaluation_output_dir=eval_dir,
            metric_key=metric_key,
            metric_k=metric_k,
            logger=logger,
            save_predictions=save_predictions,
        )

        trial.set_user_attr(
            "post_training_evaluation_score",
            final_score,
        )

        with (
            trial_dir
            / "post_training_evaluation.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "score": final_score,
                    "metric_key": metric_key,
                    "metric_k": metric_k,
                    "evaluation_split": evaluation_split,
                    "checkpoint": str(best_checkpoint),
                    "results": json_safe(final_results),
                },
                file,
                indent=2,
            )

        logger.info(
            "Optuna objective from fresh checkpoint evaluation: "
            f"{final_score:.6f}"
        )

        return final_score

    except torch.cuda.OutOfMemoryError:
        trial.set_user_attr(
            "oom",
            True,
        )

        bad_score = (
            -1.0e30
            if direction == "maximize"
            else 1.0e30
        )

        logger.exception(
            "CUDA out of memory; assigning a dominated score"
        )

        clear_cuda()

        return bad_score

    finally:
        if model is not None:
            del model

        clear_cuda()


# =============================================================================
# STUDY OUTPUTS
# =============================================================================

def save_best_config(
    *,
    study: optuna.Study,
    experiment_dir: Path,
    config_file: str,
    cli_opts: list[str],
    task: str,
) -> None:
    try:
        best_trial = study.best_trial
    except ValueError:
        return

    best_overrides = {
        normalize_path(path): value
        for path, value in best_trial.params.items()
    }

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

    summary = {
        "study_name": study.study_name,
        "trial_number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
    }

    with (
        experiment_dir / "best_result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    OmegaConf.save(
        OmegaConf.create(
            {
                "fixed_overrides": FIXED_OVERRIDES,
                "tuned_overrides": best_overrides,
                "objective_value": best_trial.value,
                "source_trial": best_trial.number,
            }
        ),
        experiment_dir / "best_overrides.yaml",
    )


def persist_study_tables(
    study: optuna.Study,
    experiment_dir: Path,
) -> None:
    try:
        study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "datetime_start",
                "datetime_complete",
                "duration",
                "params",
                "user_attrs",
                "state",
            ),
            multi_index=False,
        ).to_csv(
            experiment_dir / "all_trials.csv",
            index=False,
        )

    except Exception as error:
        print(
            "Warning: could not save all_trials.csv: "
            f"{error}"
        )


def completed_trial_callback(
    study: optuna.Study,
    frozen_trial: optuna.trial.FrozenTrial,
    *,
    experiment_dir: Path,
    config_file: str,
    cli_opts: list[str],
    task: str,
) -> None:
    persist_study_tables(
        study,
        experiment_dir,
    )

    if frozen_trial.state == TrialState.COMPLETE:
        save_best_config(
            study=study,
            experiment_dir=experiment_dir,
            config_file=config_file,
            cli_opts=cli_opts,
            task=task,
        )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Direct Optuna tuning for SGG-Benchmark with fresh "
            "post-training checkpoint evaluation."
        )
    )

    parser.add_argument(
        "--config-file",
        required=True,
    )

    parser.add_argument(
        "--task",
        choices=[
            "sgdet",
            "sgcls",
            "predcls",
        ],
        default="sgdet",
    )

    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Metric alias: mR, R, F1, zR, "
            "ng-zR, ng-R or ng-mR."
        ),
    )

    parser.add_argument(
        "--metric-key",
        default=None,
        help=(
            "Exact key returned by inference; "
            "overrides --metric."
        ),
    )

    parser.add_argument(
        "--metric-k",
        type=int,
        default=50,
        help=(
            "Evaluate the requested metric at this cutoff, e.g. mR@50. "
            "Use 0 to average all available cutoffs."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["max", "min"],
        default="max",
        help=(
            "Compatibility alias mapped to "
            "Optuna maximize/minimize."
        ),
    )

    parser.add_argument(
        "--num-trials",
        "--num-samples",
        dest="num_trials",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
        help=(
            "Approximate training images per epoch; "
            "0 uses the full loader."
        ),
    )

    parser.add_argument(
        "--evaluation-split",
        choices=["val", "test"],
        default="val",
        help=(
            "Split used by the fresh post-training evaluator. "
            "Use val for hyperparameter selection; reserve test "
            "for final reporting."
        ),
    )

    parser.add_argument(
        "--internal-validation-split",
        choices=["val", "test"],
        default="val",
        help=(
            "Split passed to run_val after every training epoch. "
            "Keep val for unbiased hyperparameter selection; use "
            "test only for an explicit diagnostic experiment."
        ),
    )

    parser.add_argument(
        "--storage-path",
        default="./hyperparameter_optimization",
    )

    parser.add_argument(
        "--experiment-name",
        "--study-name",
        dest="experiment_name",
        default="sgg_optuna",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Parallel Optuna workers in this process. Keep 1 for a "
            "single GPU. Defaults to --max-concurrent-trials or 1."
        ),
    )

    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=1,
        help=(
            "Legacy Ray option; used as Optuna n_jobs "
            "when --n-jobs is absent."
        ),
    )

    parser.add_argument(
        "--pruner",
        choices=["none", "median"],
        default="none",
        help=(
            "Default none ensures every trial reaches independent "
            "final evaluation. Median pruning uses internal "
            "per-epoch validation."
        ),
    )

    parser.add_argument(
        "--pruning-startup-trials",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--pruning-warmup-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--save-eval-predictions",
        action="store_true",
        help=(
            "Pass output folders to inference during "
            "post-training evaluation."
        ),
    )

    parser.add_argument(
        "--keep-nonbest-checkpoints",
        action="store_true",
        help=(
            "Keep superseded per-epoch best checkpoints "
            "inside every trial."
        ),
    )

    parser.add_argument(
        "--reset-study",
        action="store_true",
        help=(
            "Delete the previous SQLite study and "
            "trial directory first."
        ),
    )

    # Accepted only so existing Ray command lines do not fail.
    parser.add_argument(
        "--gpus-per-trial",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--cpus-per-trial",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--ray-temp-dir",
        default=None,
    )

    parser.add_argument(
        "--grace-period",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
    )

    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Optional fixed key/value overrides after --opts. "
            "Arguments after --opts are consumed as config overrides."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_trials < 1:
        raise ValueError(
            "--num-trials/--num-samples must be >= 1"
        )

    if args.max_epochs < 1:
        raise ValueError(
            "--max-epochs must be >= 1"
        )

    if args.max_images < 0:
        raise ValueError(
            "--max-images must be >= 0"
        )

    if args.metric_k < 0:
        raise ValueError(
            "--metric-k must be >= 0"
        )

    if args.gpus_per_trial != 1.0:
        raise ValueError(
            "This direct Optuna script currently supports "
            "one GPU per trial."
        )

    n_jobs = (
        args.n_jobs
        if args.n_jobs is not None
        else args.max_concurrent_trials
    )

    if n_jobs < 1:
        raise ValueError(
            "--n-jobs must be >= 1"
        )

    if n_jobs != 1:
        raise ValueError(
            "This script intentionally supports n_jobs=1 only: "
            "SGG-Benchmark logging and CUDA device assignment are "
            "process-global. Run separate Optuna worker processes "
            "with explicit CUDA_VISIBLE_DEVICES only after adding "
            "a multi-worker storage/device strategy."
        )

    config_file = str(
        Path(args.config_file)
        .expanduser()
        .resolve()
    )

    if not Path(config_file).is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_file}"
        )

    storage_root = (
        Path(args.storage_path)
        .expanduser()
        .resolve()
    )

    experiment_dir = (
        storage_root
        / args.experiment_name
    )

    database_path = (
        experiment_dir
        / "optuna.db"
    )

    if (
        args.reset_study
        and experiment_dir.exists()
    ):
        shutil.rmtree(
            experiment_dir
        )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        experiment_dir
        / "trials"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_user_configuration(
        config_file=config_file,
        cli_opts=args.opts,
        task=args.task,
    )

    direction = (
        "maximize"
        if args.mode == "max"
        else "minimize"
    )

    sampler = TPESampler(
        seed=args.seed,
        multivariate=True,
    )

    if args.pruner == "median":
        pruner = MedianPruner(
            n_startup_trials=args.pruning_startup_trials,
            n_warmup_steps=args.pruning_warmup_epochs,
        )
    else:
        pruner = NopPruner()

    storage_url = (
        f"sqlite:///{database_path}"
    )

    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "connect_args": {
                "timeout": 60,
            }
        },
    )

    study = optuna.create_study(
        study_name=args.experiment_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )

    study.set_user_attr(
        "config_file",
        config_file,
    )

    study.set_user_attr(
        "task",
        args.task,
    )

    study.set_user_attr(
        "metric",
        args.metric,
    )

    study.set_user_attr(
        "metric_key",
        args.metric_key,
    )

    study.set_user_attr(
        "metric_k",
        args.metric_k or "mean",
    )

    study.set_user_attr(
        "evaluation_split",
        args.evaluation_split,
    )

    study.set_user_attr(
        "internal_validation_split",
        args.internal_validation_split,
    )

    print("\nOptuna study ready")
    print(
        f"  study: {study.study_name}"
    )
    print(
        f"  database: {database_path}"
    )
    print(
        f"  direction: {direction}"
    )
    print(
        "  per-epoch run_val split: "
        f"{args.internal_validation_split}"
    )
    print(
        "  final evaluation split: "
        f"{args.evaluation_split}"
    )

    if args.metric_k:
        print(
            f"  objective cutoff: @{args.metric_k}"
        )
    else:
        print(
            "  objective cutoff: mean"
        )

    print("\nLive dashboard command:")

    print(
        "  uv run optuna-dashboard "
        f"\"sqlite:///{database_path}\" "
        "--host 127.0.0.1 --port 8080"
    )

    def callback(
        study_: optuna.Study,
        trial_: optuna.trial.FrozenTrial,
    ) -> None:
        completed_trial_callback(
            study_,
            trial_,
            experiment_dir=experiment_dir,
            config_file=config_file,
            cli_opts=args.opts,
            task=args.task,
        )

    started = time.time()

    study.optimize(
        lambda trial: objective(
            trial,
            config_file=config_file,
            cli_opts=args.opts,
            task=args.task,
            requested_metric=args.metric,
            explicit_metric_key=args.metric_key,
            metric_k=(
                args.metric_k
                or None
            ),
            direction=direction,
            max_epochs=args.max_epochs,
            max_images=args.max_images,
            experiment_dir=experiment_dir,
            evaluation_split=args.evaluation_split,
            internal_validation_split=(
                args.internal_validation_split
            ),
            save_predictions=(
                args.save_eval_predictions
            ),
            keep_nonbest_checkpoints=(
                args.keep_nonbest_checkpoints
            ),
        ),
        n_trials=args.num_trials,
        n_jobs=n_jobs,
        callbacks=[callback],
        gc_after_trial=True,
        show_progress_bar=(
            n_jobs == 1
            and sys.stdout.isatty()
        ),
    )

    persist_study_tables(
        study,
        experiment_dir,
    )

    save_best_config(
        study=study,
        experiment_dir=experiment_dir,
        config_file=config_file,
        cli_opts=args.opts,
        task=args.task,
    )

    elapsed = (
        time.time()
        - started
    )

    completed = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ]

    print("\nOptuna tuning complete")
    print(
        f"  elapsed seconds: {elapsed:.1f}"
    )
    print(
        f"  completed trials: {len(completed)}"
    )

    if completed:
        print(
            f"  best trial: {study.best_trial.number}"
        )
        print(
            f"  best value: {study.best_value}"
        )
        print(
            "  best parameters:"
        )

        for path, value in study.best_params.items():
            print(
                f"    {path}: {value}"
            )

    print(
        f"  outputs: {experiment_dir}"
    )
    print(
        f"  database: {database_path}"
    )


if __name__ == "__main__":
    main()
# Optuna Hyperparameter Optimization

## Install

```bash
uv pip install optuna optuna-dashboard sqlalchemy pandas
```

## Run optimization

From the SGG-Benchmark project root:

```bash
python tools/optimize_optuna.py \
  --config-file "$(pwd)/configs/hydra/NV3_v2/React++_Yolo12m_SGDet_v18.yaml" \
  --search-space-file "$(pwd)/configs/optuna/reactpp_search_space.yaml" \
  --task sgdet \
  --metric mR \
  --metric-k 50 \
  --mode max \
  --num-samples 30 \
  --max-epochs 5 \
  --max-images 200 \
  --internal-validation-split val \
  --evaluation-split val \
  --storage-path "$(pwd)/hyperparameter_optimization" \
  --experiment-name reactpp_optuna
```

The study is stored in:

```text
hyperparameter_optimization/reactpp_optuna/
```

## Open Optuna Dashboard

```bash
uv run optuna-dashboard \
  "sqlite:///$(pwd)/hyperparameter_optimization/reactpp_optuna/optuna.db" \
  --host 127.0.0.1 \
  --port 8080
```

Open:

```text
http://localhost:8080
```

For a remote server, create an SSH tunnel from your local machine:

```bash
ssh -N -L 8080:127.0.0.1:8080 nyquist@harry
```

Then open `http://localhost:8080` locally.



```bash
SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    # =====================================================================
    # DATA AUGMENTATION
    # =====================================================================
    #
    # Intervalli volutamente moderati: alterazioni troppo forti possono
    # modificare indizi visivi importanti per riconoscere le relazioni.
    #
    "input.brightness": {
        "type": "uniform",
        "low": 0.00,
        "high": 0.30,
    },
    "input.contrast": {
        "type": "uniform",
        "low": 0.00,
        "high": 0.30,
    },
    "input.saturation": {
        "type": "uniform",
        "low": 0.00,
        "high": 0.25,
    },
    "input.hue": {
        "type": "uniform",
        "low": 0.00,
        "high": 0.08,
    },

    # =====================================================================
    # YOLO BACKBONE / DETECTOR
    # =====================================================================
    #
    # Il parametro varia su più ordini di grandezza, quindi loguniform è
    # più appropriato di uniform.
    #
    "model.backbone.nms_thresh": {
        "type": "loguniform",
        "low": 1.0e-3,
        "high": 1.5e-1,
    },

    # =====================================================================
    # ROI HEADS / OBJECT DETECTION
    # =====================================================================
    "model.roi_heads.fg_iou_threshold": {
        "type": "uniform",
        "low": 0.50,
        "high": 0.85,
    },
    "model.roi_heads.nms": {
        "type": "uniform",
        "low": 0.25,
        "high": 0.60,
    },
    "model.roi_heads.detections_per_img": {
        "type": "choice",
        "values": [15, 25, 40, 50, 75, 100],
    },

    # =====================================================================
    # RELATION HEAD
    # =====================================================================
    "model.roi_relation_head.context_dropout_rate": {
        "type": "uniform",
        "low": 0.05,
        "high": 0.40,
    },
    "model.roi_relation_head.batch_size_per_image": {
        "type": "choice",
        "values": [16, 32, 64, 128],
    },
    "model.roi_relation_head.positive_fraction": {
        "type": "uniform",
        "low": 0.20,
        "high": 0.55,
    },
    "model.roi_relation_head.num_sample_per_gt_rel": {
        "type": "choice",
        "values": [4, 8, 16, 24, 32],
    },
    "model.roi_relation_head.add_gtbox_to_proposal_in_train": {
        "type": "choice",
        "values": [False, True],
    },

    # ATTENZIONE:
    # non abilitare questo parametro se esegui tutti i trial con
    # --task sgdet. Lo script set_task_mode() imposta use_gt_box=False
    # per SGDet e sovrascrive il valore campionato.
    #
    # "model.roi_relation_head.use_gt_box": {
    #     "type": "choice",
    #     "values": [False, True],
    # },
}
```

```bash
SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "input.brightness": {
        "type": "uniform",
        "low": 0.00,
        "high": 0.25,
    },
    "input.contrast": {
        "type": "uniform",
        "low": 0.00,
        "high": 0.25,
    },
    "input.saturation": {
        "type": "uniform",
        "low": 0.00,
        "high": 0.20,
    },
    "input.hue": {
        "type": "uniform",
        "low": 0.00,
        "high": 0.05,
    },
    "model.backbone.nms_thresh": {
        "type": "loguniform",
        "low": 5.0e-3,
        "high": 1.0e-1,
    },
    "model.roi_heads.fg_iou_threshold": {
        "type": "uniform",
        "low": 0.55,
        "high": 0.80,
    },
    "model.roi_heads.nms": {
        "type": "uniform",
        "low": 0.30,
        "high": 0.55,
    },
    "model.roi_heads.detections_per_img": {
        "type": "choice",
        "values": [25, 50, 75],
    },
    "model.roi_relation_head.context_dropout_rate": {
        "type": "uniform",
        "low": 0.10,
        "high": 0.35,
    },
    "model.roi_relation_head.batch_size_per_image": {
        "type": "choice",
        "values": [16, 32, 64],
    },
    "model.roi_relation_head.positive_fraction": {
        "type": "uniform",
        "low": 0.25,
        "high": 0.50,
    },
    "model.roi_relation_head.num_sample_per_gt_rel": {
        "type": "choice",
        "values": [8, 16, 32],
    },
    "model.roi_relation_head.add_gtbox_to_proposal_in_train": {
        "type": "choice",
        "values": [False, True],
    },
}
```

# Run hyperparameter optimization
```bash
python tools/tune_sgg_optuna.py   --config-file "/home/gino/projects/samu_test_ws/src/SGG-Benchmark/configs/hydra/NV3_v2/React++_Yolo12m_SGDet_v30.yaml"   --task sgdet   --metric mR   --metric-k 50   --mode max   --num-samples 1000   --max-epochs 5   --max-images 200   --internal-validation-split val   --evaluation-split test   --storage-path "hyperparameter_optimization"   --experiment-name reactpp_optuna_small_dataset_new_search_space

```

```bash
python tools/tune_sgg_optuna.py   --config-file "/home/gino/projects/samu_test_ws/src/SGG-Benchmark/configs/hydra/NV3_v2/React++_Yolo12m_SGDet_v30_rebalanced.yaml"   --task sgdet   --metric mR   --metric-k 50   --mode max   --num-samples 1000   --max-epochs 10   --max-images 1200   --internal-validation-split val   --evaluation-split test   --storage-path "hyperparameter_optimization"   --experiment-name reactpp_optuna_rebalanced1
```
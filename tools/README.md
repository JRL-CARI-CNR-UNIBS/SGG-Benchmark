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

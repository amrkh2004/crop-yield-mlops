# Module 2 Report: MLflow Experiment Tracking, DVC Pipeline & CI/CD Quality Gate

## 1. Overview
Module 2 establishes experiment tracking, artifact versioning, data pipeline reproducibility, and automated CI/CD quality gates for the Crop Yield Prediction MLOps service.

---

## 2. MLflow Experiment Tracking & Model Registry (`src/prodml/mlflow_tracker.py`)
The tracking system logs 6 candidate model architectures on real Kaggle crop yield data:
1. `Ridge_Baseline`: `Ridge(alpha=1.0)`
2. `Linear_Regression`: `LinearRegression()`
3. `Decision_Tree_Depth10`: `DecisionTreeRegressor(max_depth=10)`
4. `Random_Forest_Tuned`: `RandomForestRegressor(n_estimators=100, max_depth=15)`
5. `Gradient_Boosting`: `GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5)`
6. `Extra_Trees`: `ExtraTreesRegressor(n_estimators=100, max_depth=15)`

### Logged Artifacts & Metrics:
- **Parameters**: `model_type`, `n_estimators`, `learning_rate`, `max_depth`, `alpha`.
- **Metrics**: `MAE_hg_ha`, `MAE_tpha`, `RMSE`, `R2`.
- **Tags**: `experiment_type=crop_yield_pipeline`, `dataset=kaggle_crop_yield_real`, `author=mlops_team`.
- **Registry Promotion**: Top candidate model (highest R² / lowest MAE) is registered in the MLflow Model Registry (`CropYieldModel`) and promoted to stage `Production` with alias `Production`.

---

## 3. DVC Data & Pipeline Versioning (`dvc.yaml`)
DVC manages data dependencies and pipeline stages across environments using standard `python`:
```yaml
stages:
  prepare:
    cmd: python -m prodml.pipeline.prepare
    deps:
      - src/prodml/pipeline/prepare.py
      - src/prodml/data.py
      - data/raw/crop_yield_raw.csv
    outs:
      - data/prepared/train.csv
      - data/prepared/test.csv

  train:
    cmd: python -m prodml.pipeline.train_stage
    deps:
      - src/prodml/pipeline/train_stage.py
      - src/prodml/features.py
      - data/prepared/train.csv
    outs:
      - models/model.pkl

  evaluate:
    cmd: python -m prodml.pipeline.evaluate
    deps:
      - src/prodml/pipeline/evaluate.py
      - data/prepared/test.csv
      - models/model.pkl
    metrics:
      - reports/metrics.json:
          cache: false
```

---

## 4. DVC Remote Storage & Evaluator Access Guide
- **Configured S3-Compatible Remote (MinIO & AWS)**: `s3://dvcstore` (`http://localhost:9000`)
- **Data Pushed (`dvc push`)**: Model artifacts and DVC pipeline hashes pushed to self-hosted S3-compatible object storage.
- **Evaluator / Peer Review Access (No AWS Card / Credentials Required)**:
  1. **Reproducing the data:** the raw dataset is versioned with DVC only (it is not stored in Git). Launch the local stack via `docker-compose up -d` (which runs MinIO S3 container on port 9000 with pre-configured bucket `dvcstore`), then run `dvc pull` and `dvc repro`. Alternatively, download the Kaggle "Crop Yield Prediction" dataset into `data/raw/crop_yield_raw.csv`; its expected md5 is recorded in `data/raw/crop_yield_raw.csv.dvc`.
  2. **Automated Dataset Downloader**: If data needs to be verified or checked locally:
     ```bash
     python scripts/download_data.py
     ```
  3. **AWS S3 / Custom Remote Testing**: Reviewers wishing to test `dvc pull` / `dvc push` with their own AWS S3 bucket can configure a custom remote via:
     ```bash
     dvc remote add -d my_remote s3://my-custom-bucket/dvcstore
     ```

---

## 5. GitHub Actions CI/CD Quality Gate (`.github/workflows/ci-cd.yml`)
Automated pipeline executes on push and pull requests to `main`:
1. **Ruff Linting**: `ruff check .`
2. **Black Formatting Check**: `black --check .`
3. **Pytest & Coverage Gate**: Enforces `--cov-fail-under=70` coverage threshold.
4. **Docker Build & Push**: Automatically builds and pushes `amrkh2004/crop-yield-mlops:latest` to Docker Hub upon successful quality gate pass.

---

## 6. Comprehensive Data Splitting & Out-of-Distribution (OOD) Analysis

To thoroughly evaluate model generalization and prevent data leakage, four distinct train/test splitting strategies were evaluated on the real Kaggle crop yield dataset (`data/raw/crop_yield_raw.csv`) using `scripts/evaluate_splits.py`:

| Splitting Strategy | Test Samples (`n_test`) | MAE (hg/ha) | MAE (t/ha) | $R^2$ Score | Performance & Generalization Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Shuffle** (DVC Baseline) | 5,187 | **7,902 hg/ha** | **0.79 t/ha** | **0.9651** | Excellent interpolation for known countries across harvested years |
| **Temporal Split** (train $\le$ 2008, test > 2008) | 5,750 | **12,844 hg/ha** | **1.28 t/ha** | **0.9323** | Solid time-series forecasting capability over future years for known countries |
| **Ordered 80/20** (No Shuffle) | 5,187 | **51,167 hg/ha** | **5.12 t/ha** | **-0.0957** | Fails because dataset is sorted alphabetically by country, placing late-alphabet countries in test set |
| **Unseen Countries** (Group Split by `Area`) | 3,137 | **59,591 hg/ha** | **5.96 t/ha** | **-0.1399** | **Model does NOT generalize to unseen countries** |

> [!IMPORTANT]
> **Explicit Evaluation Rigor & Generalization Limit**:
> **The model does NOT generalize to unseen countries ($R^2 = -0.1399$, $\text{MAE} = 59,591\text{ hg/ha} / 5.96\text{ t/ha}$)**. 
> Because geographic `Area` is a high-cardinality categorical feature processed via Target / One-Hot Encoding, evaluating predictions on countries never observed during model training forces the pipeline to fall back to global mean crop yields. The model learns region-specific historical yields rather than universal causal relationships from weather/pesticide inputs alone. Random shuffle split ($R^2 = 0.9651$) is appropriate when serving inferences for known agricultural regions over time, whereas zero-shot prediction for new unobserved countries would require domain adaptation or macro-region embeddings.


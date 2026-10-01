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
- **Configured S3 Remote**: `s3://crop-yield-dvc-remote-store/crop_yield_dvc`
- **Data Pushed (`dvc push`)**: Pushed to remote storage.
- **Evaluator / Peer Review Access (No AWS Credentials Required)**:
  - Raw and prepared data files (`data/raw/crop_yield_raw.csv`, `data/prepared/train.csv`, `data/prepared/test.csv`) are tracked and provided directly in the codebase repository/zip.
  - Reviewers can run `dvc repro` and all test suites locally out-of-the-box without needing AWS S3 IAM credentials.
  - If a reviewer wishes to test `dvc pull` / `dvc push` with their own S3 bucket, they can configure a custom remote via:
    ```bash
    dvc remote add -d my_remote s3://my-custom-bucket/dvc-store
    ```

---

## 5. GitHub Actions CI/CD Quality Gate (`.github/workflows/ci-cd.yml`)
Automated pipeline executes on push and pull requests to `main`:
1. **Ruff Linting**: `ruff check .`
2. **Black Formatting Check**: `black --check .`
3. **Pytest & Coverage Gate**: Enforces `--cov-fail-under=70` coverage threshold.
4. **Docker Build & Push**: Automatically builds and pushes `amrkh2004/crop-yield-mlops:latest` to Docker Hub upon successful quality gate pass.

---

## 6. Data Splitting Leakage & Out-of-Distribution Analysis

During pipeline optimization, two distinct data splitting strategies were evaluated on the Kaggle Crop Yield dataset (`data/raw/crop_yield_raw.csv`):

### A. Sequential Country Split (Unshuffled `df.iloc[:split_idx]`)
* **Behavior**: The raw Kaggle dataset is sorted alphabetically by country (`Area`). Splitting sequentially without shuffling places countries starting with P through Z (e.g., *Pakistan, Poland, Qatar, Saudi Arabia, Senegal, Zimbabwe*) exclusively in the test set.
* **Impact**: Categorical feature encoders (`TargetEncoder` and `OneHotEncoder`) encounter unseen countries in the test set, falling back to global mean yields.
* **Metrics**: $R^2 = -0.0957$, $\text{MAE} = 51,167.41 \text{ hg/ha} \; (5.11 \text{ t/ha})$.
* **Insight**: Highlights out-of-distribution (OOD) generalization performance on entirely unobserved geographic regions.

### B. Shuffled Random Split (`train_test_split(df, test_size=0.2, random_state=42, shuffle=True)`)
* **Behavior**: Samples across all countries and years are randomly assigned between training (80%) and test (20%) sets.
* **DVC Baseline Metrics**: $R^2 = 0.9651$, $\text{MAE} = 7,902.35 \text{ hg/ha} \; (0.79 \text{ t/ha})$.
* **Best Registered Architecture (Extra Trees)**: $R^2 = 0.9824$, $\text{MAE} = 4,848.02 \text{ hg/ha} \; (0.48 \text{ t/ha})$.
* **Insight**: Evaluates model interpolation capacity and yield prediction precision across temporal variations in known agricultural regions.


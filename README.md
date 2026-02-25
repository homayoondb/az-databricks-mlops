# az-mlops

Lightweight MLOps scaffolding for Databricks projects. Adds production-grade MLOps to any ML project in ~12 files — no complex templates, no 20-question wizards.

Built for AWS Databricks with Unity Catalog and GitHub Actions.

## Why?

Most ML teams have working models but no production pipeline. Setting up MLOps from scratch means figuring out Databricks Asset Bundles, model validation, CI/CD workflows, and deployment promotion — all from blank files.

`az-mlops` does it in one command. You point it at your training script, answer 3 prompts, and get a production-ready pipeline.

## Installation

```bash
# Install directly from GitHub (recommended)
pip install git+https://github.com/<org>/az-mlops.git

# Or clone and install locally
git clone https://github.com/<org>/az-mlops.git
pip install ./az-mlops
```

After installation, the `az-mlops` command is available globally.

## Quick start

```bash
# Add MLOps to an existing project (the most common case)
cd my_ml_project
az-mlops init

# Or create a new project from scratch
az-mlops new my_project \
  --staging-url https://staging.cloud.databricks.com \
  --prod-url https://prod.cloud.databricks.com
```

### What `az-mlops init` looks like

```
$ cd my_ml_project
$ az-mlops init

Project name [my_ml_project]:
Staging workspace URL: https://staging.cloud.databricks.com
Prod workspace URL: https://prod.cloud.databricks.com

  Found notebooks/scripts in your project:
    1. notebooks/train_model.py
    2. notebooks/exploration.ipynb
    3. src/preprocess.py

  Training notebook/script (number or path) [training/notebooks/Train.py]: 1
  Include batch inference job? [Y/n]: n

  Created .gitignore
  Created databricks.yml
  Created resources/training-job.yml
  Created mlops/config.py
  Created mlops/validation.py
  ...
  Created GETTING_STARTED.md

Done! Next steps are in GETTING_STARTED.md
```

The CLI scans your project for `.py` and `.ipynb` files so you can pick your training script by number instead of typing the path. After that, open `GETTING_STARTED.md` — it's a 4-step checklist personalized to your project.

## What gets generated

```
my_project/
├── .gitignore
├── databricks.yml              # Bundle config: dev/staging/prod targets
├── GETTING_STARTED.md          # 4-step onboarding checklist
├── resources/
│   ├── training-job.yml        # Train → Validate → Deploy workflow
│   └── inference-job.yml       # Scheduled batch inference (optional)
├── mlops/
│   ├── __init__.py
│   ├── config.py               # Project config (model name, catalog, schema)
│   ├── validation.py           # Model quality thresholds (customizable)
│   ├── run_validation.py       # Notebook: runs mlflow.evaluate() with thresholds
│   ├── deploy.py               # UC alias promotion logic
│   └── run_deploy.py           # Notebook: runs promotion (challenger → champion)
└── .github/workflows/
    ├── ci.yml                  # PR: bundle validate + pytest
    └── cd.yml                  # Deploy: staging on main, prod on tag
```

Existing code is never modified — only new files are added alongside it.

## Try it on a real (messy) project

The `examples/messy-ml-project/` directory is a deliberately sloppy ML project — scattered notebooks, dead code, pickle files. Run the demo to see `az-mlops` add MLOps to it without touching a single existing file:

```bash
bash examples/demo.sh
```

Or do it manually:

```bash
cd examples/messy-ml-project
python notebooks/train_model_v3_FINAL.py   # verify it works
az-mlops init                               # add MLOps
cat GETTING_STARTED.md                      # see what to do next
```

## Commands

### `az-mlops init`

Add MLOps scaffolding to the current directory.

```bash
# Interactive (discovers your notebooks, prompts for choices)
az-mlops init

# Non-interactive (CI-friendly)
az-mlops init \
  --project-name my_project \
  --staging-url https://staging.cloud.databricks.com \
  --prod-url https://prod.cloud.databricks.com \
  --training-notebook notebooks/train.py \
  --skip-inference
```

Options:
- `--training-notebook` — path to your training script (prompted interactively if omitted)
- `--inference-notebook` — path to your inference script
- `--skip-inference` — skip batch inference job entirely
- `--with-dqx` — include DQX data quality checks
- `--overwrite` — replace existing generated files

### `az-mlops new <name>`

Create a new project directory with MLOps scaffolding.

```bash
az-mlops new my_project \
  --staging-url https://staging.cloud.databricks.com \
  --prod-url https://prod.cloud.databricks.com
```

### `az-mlops add dqx`

Add DQX data quality checks to an existing `az-mlops` project.

```bash
az-mlops add dqx
```

Generates `mlops/dqx_checks.py` and `resources/dqx-job.yml`. When added at init time with `--with-dqx`, the training job automatically depends on the data quality check passing.

## DQX integration (optional)

DQX data quality checks are opt-in, either at init time or later:

```bash
# At project creation
az-mlops init --with-dqx

# Or add later to an existing project
az-mlops add dqx
```

When enabled, a `DataQuality` task runs before training and blocks it if checks fail. This keeps the default experience minimal while letting teams adopt data quality incrementally.

## Deployment flow

1. **PR opened** → `ci.yml` validates the bundle against staging and prod, runs `pytest`
2. **Merge to main** → `cd.yml` deploys to the staging workspace
3. **Push a tag (`v*`)** → `cd.yml` deploys to prod

### Required GitHub secrets

| Secret | Purpose |
|--------|---------|
| `STAGING_WORKSPACE_TOKEN` | Databricks PAT for staging workspace |
| `PROD_WORKSPACE_TOKEN` | Databricks PAT for prod workspace |

## Training pipeline

The generated training job runs three tasks in sequence:

1. **Train** — runs your training notebook, logs to MLflow, registers the model in Unity Catalog
2. **ModelValidation** — evaluates the model against configurable thresholds (edit `mlops/validation.py`)
3. **ModelDeployment** — promotes the model via UC aliases (`challenger` → `champion`)

If `--with-dqx` is enabled, a **DataQuality** task runs first and gates training.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Customization

- **Training notebook** — change `notebook_path` in `resources/training-job.yml`
- **Cluster config** — edit node type, workers, Spark version in resource YAMLs
- **Validation thresholds** — edit `mlops/validation.py` (metric names, threshold values)
- **Schedule** — edit the `schedule` block in any resource YAML (cron expression)
- **Model promotion logic** — edit `mlops/deploy.py`
- **DQX rules** — edit `mlops/dqx_checks.py` (add/remove column-level checks)

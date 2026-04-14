# az-databricks-mlops

Lightweight MLOps toolkit for Databricks projects. Two main capabilities:

1. **`adm document`** — Review any repository (classic ML, LLM, RAG, agentic, or hybrid) and generate a prioritized MLOps/LLMOps improvement report using Databricks Model Serving
2. **`adm init`** — Add production-grade scaffolding to any ML or LLM project — choose **classic ML** or **LLMOps** during setup

During `adm init`, you pick a project type:

| Project Type | Pipeline | Use For |
|---|---|---|
| `classic_ml` | Train → Validate → Deploy | scikit-learn, XGBoost, PyTorch, etc. |
| `llmops` | AgentDev → AgentEval → AgentDeploy | LLM agents, RAG apps, any framework |

Both use Unity Catalog model registration and alias-based promotion (challenger → champion). The LLMOps scaffold is **framework-agnostic** — your agent script just needs to expose an `agent` callable or `predict` function.

The published package name is `az-databricks-mlops`. The CLI command is `adm`.

Built for Databricks with Unity Catalog.

## Installation

```bash
pip install git+https://github.com/homayoondb/az-databricks-mlops.git
```

## Repository Review (`adm document`)

Point `adm document` at any local folder or remote Git URL and it generates a detailed Markdown report covering MLOps readiness, LLMOps best practices, architecture quality, and prioritized next steps.

It automatically classifies the repo (classic ML, LLM app, RAG, agentic, data pipeline, hybrid) and applies the right evaluation framework — classic MLflow for traditional ML, MLflow 3 tracing and `mlflow.genai.evaluate()` for LLM/agent repos.

```bash
# Review the current directory
adm document

# Review a local project
adm document --source path/to/project

# Review a remote Git repository
adm document --source https://github.com/org/repo.git

# Specify output path and endpoint
adm document --source . --output review.md --endpoint databricks-gpt-5-4
```

What you see while it runs:

```
adm document — generating repository review

  [1/7] Connecting to Databricks workspace...
  [2/7] Resolving repository source (endpoint: databricks-gpt-5-4)...
  [3/7] Scanning files in /path/to/project
  [4/7] Collected 22 files (23,471 chars), omitted 10
         data/houses.csv
         mlops/__init__.py, config.py, deploy.py, ...
         notebooks/exploration.py, train_model_v3_FINAL.py
         ...

         Skipped 10 files:
           model.pkl (binary or unsupported extension)
           .databricks/bundle/dev/... (ignored irrelevant directory)
  [5/7] Building review prompt...
  [6/7] Sending to databricks-gpt-5-4 for review...
         Generating... 5,429 chars received (20s)
         Generating... 10,026 chars received (35s)
         Done — 14,302 chars received in 49s
  [7/7] Writing review to project-adm-review.md

  Done!

  Review:   /path/to/project-adm-review.md
  Endpoint: databricks-gpt-5-4
  Files:    22 included, 10 omitted
  Chars:    23,471
```

### What the review covers

The generated report includes:

- **Repository classification** — classic ML, LLM app, RAG, agentic, data pipeline, or hybrid
- **Executive summary** — what's there, what's missing
- **Weighted scorecard** — architecture, reproducibility, testing, security, observability, and ML/LLM-specific dimensions (scored 0-5)
- **Prioritized findings** — grouped into Now, Next, Later with evidence, effort, and confidence
- **Step-by-step improvement plan**

For classic ML repos it evaluates against MLflow tracking, Unity Catalog model lifecycle, validation gates, and deployment promotion. For LLM/agentic repos it evaluates against MLflow 3 tracing, `mlflow.genai.evaluate()`, scorers/LLM judges, production monitoring, and human feedback loops.

### Smart filtering

- Respects `.gitignore` rules
- Skips binary files, build artifacts, `mlruns/`, `wandb/`, logs, `__pycache__/`, `.databricks/`
- Truncates large files to stay within prompt budget
- Shows exactly which files were included and which were skipped (and why)

### Endpoint auto-selection and fallback

If you don't specify `--endpoint`, the tool picks the best available Databricks Model Serving endpoint automatically (prefers OpenAI models, then Google, then Anthropic). If an endpoint doesn't respond within 10 seconds, it falls back to the next one.

### Options

- `--source` — local path or Git URL to review (defaults to current directory)
- `--output` — path for the generated Markdown document
- `--endpoint` — specific Databricks Model Serving endpoint (auto-selects if omitted)
- `--max-file-chars` — max characters per file (default: 120,000)
- `--max-total-chars` — max total characters across all files (default: 2,400,000)

---

## Scaffolding (`adm init` / `adm run`)

### Why?

Most ML teams have working models but no production pipeline. Setting up MLOps from scratch means figuring out Databricks Asset Bundles, model registration, validation, and deployment promotion — all from blank files.

`adm init` does it in one command. You point it at your training script, answer a few prompts, and get a production-ready pipeline. **Your existing training code stays untouched** — the tool wraps it with automatic MLflow tracking, model registration, and validation.

### The complete flow

```
adm init    →    adm run    →    open the experiment URL
 (prompts)     (deploy + train)    (MLflow tracking in Databricks)
```

#### Step 1 — `adm init`

```
$ cd my_ml_project
$ adm init

Project name [my_ml_project]:
Staging workspace URL: https://xxx.cloud.databricks.com
Prod workspace URL (enter to skip):

  Found notebooks/scripts in your project:
    1. notebooks/train_model.py
    2. notebooks/exploration.ipynb
    3. src/preprocess.py

  Training notebook/script (number or path) [train.py]: 1
  Include batch inference job? [Y/n]: n

  Created databricks.yml
  Created resources/training-job.yml
  Created mlops/run_training.py
  Created mlops/config.py
  Created mlops/validation.py
  ...

Running `databricks bundle validate`...
  Bundle is valid.

Next: run `adm run` to deploy and start your first training job.
```

`adm init` supports three layers of defaults:

1. CLI flags win if you pass them explicitly
2. `adm.yml` values become the prompt defaults if present
3. If `adm.yml` leaves a field blank, `adm` falls back to helpful built-in defaults like the current folder name, notebook discovery, and any existing `databricks.yml` values

#### Step 2 — run the training job

```bash
adm run              # deploys to dev (default)
adm run --target staging
```

Or run the standard Databricks commands directly:

```bash
databricks bundle deploy -t dev
databricks bundle run model_training_job -t dev
```

### What the training pipeline does

The job runs three tasks in sequence:

```
Train  →  ModelValidation  →  ModelDeployment
```

1. **Train** — wraps your training script with `mlflow.autolog()`, logs metrics/params/model automatically, registers the model in Unity Catalog as `challenger`
2. **ModelValidation** — evaluates the challenger against configurable thresholds (edit `mlops/validation.py` to set your metrics)
3. **ModelDeployment** — if validation passes, promotes the model to `champion` in Unity Catalog

If you added DQX, a **DataQuality** task runs first and blocks training if data checks fail.

### What gets generated

```
my_project/
├── .gitignore
├── databricks.yml              # Bundle config: dev/staging/prod targets
├── resources/
│   ├── training-job.yml        # Train → Validate → Deploy workflow
│   └── inference-job.yml       # Scheduled batch inference (optional)
├── mlops/
│   ├── __init__.py
│   ├── config.py               # Project config (model name, catalog, schema)
│   ├── run_training.py         # Wraps YOUR script with MLflow + UC registration
│   ├── validation.py           # Model quality thresholds (customizable)
│   ├── run_validation.py       # Runs mlflow.evaluate() with thresholds
│   ├── deploy.py               # UC alias promotion logic
│   ├── run_deploy.py           # Runs promotion (challenger → champion)
│   └── run_inference.py        # Loads champion model, scores input table
└── [existing ML code untouched]
```

**Your existing code is never modified** — only new files are added alongside it.

### How it works (zero code changes)

`mlops/run_training.py` wraps your existing training script automatically:

1. Sets up the MLflow experiment
2. Enables `mlflow.autolog()` — captures model, metrics, and parameters
3. Executes your training script (unchanged)
4. Registers the logged model in Unity Catalog
5. Sets the "challenger" alias for validation

Your training script doesn't need `import mlflow` or any modifications. The wrapper handles everything.

### LLMOps scaffolding (`--project-type llmops`)

For LLM/RAG/agentic projects, `adm init --project-type llmops` generates:

```
my_agent/
├── databricks.yml              # Bundle config: dev/staging/prod targets
├── resources/
│   ├── agent-job.yml           # AgentDev → AgentEval → AgentDeploy workflow
│   └── agent-serve-job.yml     # Batch agent serving (optional)
├── llmops/
│   ├── config.py               # Project config (model name, catalog, eval dataset)
│   ├── run_agent_dev.py        # Wraps YOUR agent with MLflow tracing + UC registration
│   ├── scorers.py              # Evaluation scorers (rule-based + LLM judge stubs)
│   ├── run_agent_eval.py       # Runs mlflow.genai.evaluate() with scorers
│   ├── deploy.py               # UC alias promotion logic
│   ├── run_agent_deploy.py     # Runs promotion (challenger → champion)
│   └── run_agent_serve.py      # Loads champion agent, batch invocation
└── [your agent code untouched]
```

**Agent script contract:** Your script must expose either an `agent` variable (any callable: `str → str`) or a `predict(model_input)` function. Any framework works — LangChain, LangGraph, custom code, etc.

The LLMOps pipeline:

1. **AgentDev** — wraps your agent with `mlflow.tracing.enable()`, logs a sample invocation trace, registers the agent in Unity Catalog as `challenger`
2. **AgentEval** — runs `mlflow.genai.evaluate()` with your scorers (edit `llmops/scorers.py` to add LLM judges, safety checks, etc.)
3. **AgentDeploy** — if evaluation passes, promotes the agent to `champion` in Unity Catalog

---

## All Commands

| Command | Description |
|---|---|
| `adm document` | Review a repo and generate a prioritized improvement report |
| `adm init` | Add MLOps scaffolding to the current directory |
| `adm run` | Deploy the bundle and start the training job |
| `adm trigger` | Re-run a deployed job via SDK (no CLI needed, works from notebooks) |
| `adm clean` | Remove all generated files, keep your code |
| `adm new <name>` | Create a new project directory with scaffolding |
| `adm add dqx` | Add DQX data quality checks to an existing project |

### `adm trigger`

Re-run an already-deployed training job via the Databricks SDK. No bundle redeploy, no CLI required — works from notebooks and automation scripts.

```bash
adm trigger              # triggers the dev job
adm trigger --target staging
```

### Optional `adm.yml`

If you scaffold multiple projects into the same workspace, keep an `adm.yml` in the repo root:

```yaml
project_name:

databricks:
  staging_url: https://your-workspace.cloud.databricks.com
  prod_url: ""
  catalog_name: my_catalog
  schema_name: my_schema

training:
  training_notebook:
  skip_inference: false

options:
  with_dqx: false
```

## Notebook Usage

Install the package inside a Databricks notebook:

```python
%pip install git+https://github.com/homayoondb/az-databricks-mlops.git
```

**Trigger a training job** (works with ambient workspace credentials):

```python
from az_databricks_mlops import run_training_job

run_training_job("dev-my-project-model-training-job")
```

**Generate a repository review**:

```python
from az_databricks_mlops import review_repository

artifact = review_repository(
    source="/tmp/my-repo",
    output_path="/tmp/review.md",
)
```

Not all CLI commands work in notebook environments. `trigger` and `document` work (SDK and git-based). `init`, `run`, and `clean` require the `databricks` CLI which is not available in notebook runtimes.

## Try it on real (messy) projects

### Classic ML example

The `examples/messy-ml-project/` directory is a deliberately sloppy ML project — scattered notebooks, dead code, pickle files:

```bash
cd examples/messy-ml-project
adm document                                    # generate an improvement report
adm init                                        # add MLOps scaffolding (classic_ml)
adm run                                         # deploy + run, get experiment URL
```

### Agentic example

The `examples/messy-agent-project/` directory is a messy Q&A agent — hardcoded responses, unused tools, no evaluation:

```bash
cd examples/messy-agent-project
adm document                                    # generate an improvement report
adm init                                        # add LLMOps scaffolding (llmops via adm.yml)
adm run                                         # deploy + run agent pipeline
```

## Customization

### Classic ML (`mlops/`)
- **Validation thresholds** — edit `mlops/validation.py` (metric names, threshold values)
- **Model promotion logic** — edit `mlops/deploy.py`
- **DQX rules** — edit `mlops/dqx_checks.py` (add/remove column-level checks)

### LLMOps (`llmops/`)
- **Evaluation scorers** — edit `llmops/scorers.py` (add LLM judges, safety checks, custom rule scorers)
- **Evaluation dataset** — set `EVAL_DATASET_TABLE` in `llmops/config.py` to a UC table with `inputs` and `expected_response` columns
- **Agent promotion logic** — edit `llmops/deploy.py`

### Both
- **Cluster config** — edit node type, workers, Spark version in resource YAMLs
- **Schedule** — edit the `schedule` block in any resource YAML (cron expression)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

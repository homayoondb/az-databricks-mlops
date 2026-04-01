# Example: Messy ML Project

This is a deliberately messy ML project — the kind you'd find in real life. A house price prediction model with scattered notebooks, dead code, and zero MLOps.

## What's here (before `adm init`)

```
messy-ml-project/
├── data/houses.csv                       # 20-row toy dataset
├── notebooks/
│   ├── train_model_v3_FINAL.py           # The "real" training script
│   └── exploration.py                    # Quick data exploration
├── src/preprocess.py                     # Preprocessing utils (not used by training)
├── old_stuff/
│   ├── train_v1_DONT_USE.py              # Ahmed's old linear regression
│   └── notes.txt                         # Meeting notes about needing MLOps
├── predict.py                            # Quick manual prediction script (not used by adm)
└── requirements.txt
```

## Run the demo

```bash
# 1. Verify the training script works
cd examples/messy-ml-project
pip install -r requirements.txt
python notebooks/train_model_v3_FINAL.py

# 2. Add MLOps scaffolding
adm init \
  --project-name house_prices \
  --staging-url https://your-staging.cloud.databricks.com \
  --prod-url https://your-prod.cloud.databricks.com \
  --training-notebook notebooks/train_model_v3_FINAL.py \
  --skip-inference

# 3. See what got added
find . -type f | sort

# 4. Check that the wrapper points at your training script
rg -n "train_model_v3_FINAL|run_training" mlops/run_training.py resources/training-job.yml
```

## What you get after `adm init`

```
messy-ml-project/
├── [all original files untouched]
│
├── databricks.yml              ← NEW: bundle config (dev/staging/prod)
├── resources/
│   ├── training-job.yml        ← NEW: Train → Validate → Deploy workflow
│   └── inference-job.yml       ← NEW: optional batch inference job
├── mlops/
│   ├── __init__.py             ← NEW: package marker
│   ├── config.py               ← NEW: project config
│   ├── run_training.py         ← NEW: wrapper around your training script
│   ├── validation.py           ← NEW: model quality gates
│   ├── run_validation.py       ← NEW: validation notebook
│   ├── deploy.py               ← NEW: promotion logic
│   ├── run_deploy.py           ← NEW: deployment notebook
│   └── run_inference.py        ← NEW: generic inference wrapper
└── notebooks/
    └── run_pipeline.py         ← NEW: interactive pipeline runner
```

None of the original files were modified. The generated training job calls `mlops/run_training.py`, and that wrapper references `notebooks/train_model_v3_FINAL.py`. If you enable batch inference, `adm` generates its own generic `mlops/run_inference.py`; it does not wire in the example's `predict.py`.

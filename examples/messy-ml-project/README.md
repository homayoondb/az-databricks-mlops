# Example: Messy ML Project

This is a deliberately messy ML project — the kind you'd find in real life. A house price prediction model with scattered notebooks, dead code, and zero MLOps.

## What's here (before az-mlops)

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
├── predict.py                            # Quick manual prediction script
└── requirements.txt
```

## Run the demo

```bash
# 1. Verify the training script works
cd examples/messy-ml-project
pip install -r requirements.txt
python notebooks/train_model_v3_FINAL.py

# 2. Add MLOps scaffolding
az-mlops init \
  --project-name house_prices \
  --staging-url https://your-staging.cloud.databricks.com \
  --prod-url https://your-prod.cloud.databricks.com \
  --training-notebook notebooks/train_model_v3_FINAL.py \
  --skip-inference

# 3. See what got added
find . -type f | sort

# 4. Read the personalized onboarding guide
cat GETTING_STARTED.md
```

## What you get after az-mlops init

```
messy-ml-project/
├── [all original files untouched]
│
├── databricks.yml              ← NEW: bundle config (dev/staging/prod)
├── GETTING_STARTED.md          ← NEW: 4-step onboarding checklist
├── resources/
│   └── training-job.yml        ← NEW: points to your train_model_v3_FINAL.py
├── mlops/
│   ├── config.py               ← NEW: project config
│   ├── validation.py           ← NEW: model quality gates
│   ├── run_validation.py       ← NEW: validation notebook
│   ├── deploy.py               ← NEW: promotion logic
│   └── run_deploy.py           ← NEW: deployment notebook
└── .github/workflows/
    ├── ci.yml                  ← NEW: PR validation
    └── cd.yml                  ← NEW: auto-deploy on merge
```

None of the original files were modified. The training job YAML directly references `notebooks/train_model_v3_FINAL.py`.

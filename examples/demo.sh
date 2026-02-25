#!/usr/bin/env bash
# Demo: add MLOps to the messy-ml-project example
# Run from the az-mlops repo root: bash examples/demo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$SCRIPT_DIR/messy-ml-project"
WORK_DIR=$(mktemp -d)

echo "=== Copying messy-ml-project to temp dir ==="
cp -r "$EXAMPLE_DIR"/* "$WORK_DIR/"
cd "$WORK_DIR"

echo ""
echo "=== Step 1: This is what the developer's project looks like ==="
find . -type f | sort
echo ""

echo "=== Step 2: Run the training script (it works!) ==="
python notebooks/train_model_v3_FINAL.py
echo ""

echo "=== Step 3: Add MLOps with az-mlops ==="
az-mlops init \
  --project-name house_prices \
  --staging-url https://staging.cloud.databricks.com \
  --prod-url "" \
  --training-notebook notebooks/train_model_v3_FINAL.py \
  --skip-inference
echo ""

echo "=== Step 4: Here's what the project looks like now ==="
find . -type f -not -path './.git/*' -not -name 'model.pkl' | sort
echo ""

echo "=== Step 5: Training wrapper references their actual notebook ==="
grep "train_model_v3_FINAL" mlops/run_training.py
echo ""

echo "=== Step 6: Training job uses the wrapper (not user's script directly) ==="
grep "notebook_path.*run_training" resources/training-job.yml
echo ""

echo "=== Done! Temp dir: $WORK_DIR ==="
echo "Run 'rm -rf $WORK_DIR' to clean up."

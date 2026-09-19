#!/usr/bin/env bash

# run this in a terminal: ./scripts/bootstrap_onyxia.sh

set -euo pipefail

REPO="$HOME/work/self-supervised-visual-representation-learning"
S3_BASE="s3://lachqar/self-supervised-visual-representation-learning"

echo
echo "=========================================="
echo "          ONYXIA ENVIRONMENT SETUP"
echo "=========================================="
echo


# ============================================================
# 1. Go to repository
# ============================================================

echo "[1/5] Opening project repository..."

cd "$REPO"

echo "Repository:"
pwd
echo


# ============================================================
# 2. Pull latest GitHub version
# ============================================================

echo "[2/5] Pulling latest code from GitHub..."

git pull --ff-only

echo
echo "Git repository up to date."
echo


# ============================================================
# 3. Install Python dependencies
# ============================================================

echo "[3/5] Installing project dependencies..."

python -m pip install -r requirements.txt

echo
echo "Dependencies ready."
echo


# ============================================================
# 4. Check PyTorch / CUDA / GPU
# ============================================================

echo "[4/5] Checking PyTorch and GPU..."

python - <<'PY'
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )
PY

echo


# ============================================================
# 5. Check S3 access
# ============================================================

echo "[5/5] Checking S3 access..."

if [[ -z "${AWS_ENDPOINT_URL:-}" ]]; then
    echo
    echo "ERROR: AWS_ENDPOINT_URL is not defined."
    exit 1
fi

if aws s3 ls \
    "$S3_BASE/" \
    --endpoint-url "$AWS_ENDPOINT_URL" \
    >/dev/null 2>&1
then

    echo "S3 access OK."

else

    echo
    echo "ERROR: S3 authentication failed."
    echo "You may need to restart the Onyxia service."
    exit 1
fi


echo
echo "=========================================="
echo "             ENVIRONMENT READY"
echo "=========================================="
echo

#!/bin/bash
set -e

DATASET="olgagmiufana1/fragrantica-com-fragrance-dataset"
OUT_DIR="../../datasets/fragrantica"

# ---- Load env file ----
if [ ! -f .env ]; then
  echo "❌ .env file not found"
  exit 1
fi

set -a
source .env
set +a

# ---- Validate ----
if [[ -z "$KAGGLE_USERNAME" || -z "$KAGGLE_KEY" ]]; then
  echo "❌ KAGGLE_USERNAME or KAGGLE_KEY missing in .env"
  exit 1
fi

# ---- Create kaggle.json (temporary) ----
mkdir -p ~/.kaggle "$OUT_DIR"

cat > ~/.kaggle/kaggle.json <<EOF
{
  "username": "$KAGGLE_USERNAME",
  "key": "$KAGGLE_KEY"
}
EOF

chmod 600 ~/.kaggle/kaggle.json

# ---- Download ----
kaggle datasets download -d "$DATASET" -p "$OUT_DIR" --unzip

echo "✅ Download complete"

# ---- Cleanup (recommended) ----
rm ~/.kaggle/kaggle.json

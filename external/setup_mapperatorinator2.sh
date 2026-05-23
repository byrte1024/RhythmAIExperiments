#!/bin/bash
# Setup Mapperatorinator2 for benchmarking
# Run from the repo root: bash external/setup_mapperatorinator2.sh

set -e

INSTALL_DIR="${1:-/home/drore/repos/Mapperatorinator2}"

echo "=== Setting up Mapperatorinator2 ==="
echo "Install dir: $INSTALL_DIR"

# Clone
if [ -d "$INSTALL_DIR" ]; then
    echo "Already cloned, pulling latest..."
    cd "$INSTALL_DIR" && git pull
else
    echo "Cloning..."
    git clone https://github.com/Tiger14n/Mapperatorinator2.git "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# Create venv with Python 3.10 if available
PYTHON310=$(which python3.10 2>/dev/null || echo "")
if [ -z "$PYTHON310" ]; then
    echo "WARNING: python3.10 not found. Mapperatorinator requires Python 3.10."
    echo "Trying system python..."
    PYTHON310="python3"
fi

if [ ! -d ".venv" ]; then
    echo "Creating venv with $PYTHON310..."
    $PYTHON310 -m venv .venv
fi

echo "Activating venv..."
source .venv/bin/activate

echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "To benchmark:"
echo "  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.benchmark_external \\"
echo "      --backend mapperatorinator \\"
echo "      --backend-path $INSTALL_DIR \\"
echo "      --dataset taiko2_v1 \\"
echo "      --fraction 0.05 \\"
echo "      --device cuda \\"
echo "      --experiment-dir osu/taiko2/experiments/018-baselines"

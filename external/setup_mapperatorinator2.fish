#!/usr/bin/env fish
# Setup Mapperatorinator2 for benchmarking
# Run: fish external/setup_mapperatorinator2.fish [install_dir]

set INSTALL_DIR (test -n "$argv[1]"; and echo $argv[1]; or echo /home/drore/repos/Mapperatorinator2)

echo "=== Setting up Mapperatorinator2 ==="
echo "Install dir: $INSTALL_DIR"

# Clone
if test -d "$INSTALL_DIR"
    echo "Already cloned, pulling latest..."
    cd "$INSTALL_DIR"; and git pull
else
    echo "Cloning..."
    git clone https://github.com/Tiger14n/Mapperatorinator2.git "$INSTALL_DIR"
end

cd "$INSTALL_DIR"

# Create venv — try python3.10 first
set PYTHON310 (which python3.10 2>/dev/null)
if test -z "$PYTHON310"
    echo "WARNING: python3.10 not found. Mapperatorinator requires Python 3.10."
    echo "Trying system python..."
    set PYTHON310 python3
end

if not test -d ".venv"
    echo "Creating venv with $PYTHON310..."
    $PYTHON310 -m venv .venv
end

echo "Activating venv..."
source .venv/bin/activate.fish

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

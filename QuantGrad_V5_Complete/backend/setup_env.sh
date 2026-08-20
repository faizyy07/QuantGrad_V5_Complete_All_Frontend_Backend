#!/bin/bash
# setup_env.sh — One-time environment setup for QuantGrad v4
# Run: bash setup_env.sh
# This creates a venv so you NEVER need to reinstall packages again.

set -e
VENV_DIR="$(pwd)/venv"

echo "==================================================="
echo " QuantGrad v4 — One-Time Environment Setup"
echo "==================================================="

# Check Python
python3 --version || { echo "Python 3 not found!"; exit 1; }

# Create venv if not already present
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment already exists — skipping creation."
fi

# Activate
echo "[2/4] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "[3/4] Upgrading pip..."
pip install --upgrade pip --quiet

# Install all packages
echo "[4/4] Installing packages from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "==================================================="
echo " Setup complete!"
echo ""
echo " To activate the environment each session:"
echo "   source venv/bin/activate"
echo ""
echo " Then run:"
echo "   python verify.py"
echo "   python macro_fetcher.py"
echo "   python trainer_v3.py --quick"
echo "   python server.py"
echo " Open: http://localhost:8000"
echo "==================================================="

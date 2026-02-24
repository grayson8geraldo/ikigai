#!/bin/bash
# IKIGAI Trading Bot — Quick Setup Script
# Usage: ./setup.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     IKIGAI Trading Bot — Setup                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found."
    echo "Install it with: brew install python@3.12"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[OK] Python $PYTHON_VERSION found"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[...] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

# Activate
source venv/bin/activate
echo "[OK] Virtual environment activated"

# Install dependencies
echo "[...] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "[OK] Dependencies installed"

# Copy .env if needed
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[OK] .env created from .env.example"
    echo ""
    echo "     Edit .env to add your Bybit API keys (optional for paper trading)"
else
    echo "[OK] .env already exists"
fi

# Create log directory
mkdir -p logs data

echo ""
echo "════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  To start the bot:"
echo ""
echo "    source venv/bin/activate"
echo "    python main.py                 # one scan"
echo "    python main.py --loop --trade  # continuous auto-trade (paper)"
echo "    python main.py --status        # check positions"
echo ""
echo "════════════════════════════════════════════════════"

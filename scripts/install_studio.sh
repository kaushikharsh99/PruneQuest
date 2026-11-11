#!/usr/bin/env bash
# PruneQuest Studio Bootstrap Installer for Linux & macOS
# Sets up an isolated Python environment and installs PruneQuest Studio.

set -euo pipefail

echo "=========================================================="
echo "🌱 PruneQuest Studio Bootstrap Installer"
echo "=========================================================="

# 1. Check Python version
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        VERSION=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$("$cmd" -c 'import sys; print(sys.version_info.major)')
        MINOR=$("$cmd" -c 'import sys; print(sys.version_info.minor)')
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            echo "✔ Found compatible Python $VERSION ($PYTHON_CMD)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python 3.10+ is required to install PruneQuest Studio."
    echo "Please install Python 3.10 or higher and re-run this script."
    exit 1
fi

# 2. Setup isolated directory
INSTALL_DIR="${HOME}/.prunequest/studio-env"
echo "Creating isolated virtual environment at ${INSTALL_DIR}..."
mkdir -p "$(dirname "$INSTALL_DIR")"
"$PYTHON_CMD" -m venv "$INSTALL_DIR"

# 3. Upgrade pip and install prunequest with studio extra
echo "Installing PruneQuest Studio..."
"$INSTALL_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
if [ -f "pyproject.toml" ]; then
    "$INSTALL_DIR/bin/python" -m pip install ".[studio]"
else
    "$INSTALL_DIR/bin/python" -m pip install "prunequest[studio]"
fi

# 4. Link launcher binary to ~/.local/bin if available
BIN_DIR="${HOME}/.local/bin"
mkdir -p "$BIN_DIR"
ln -sf "$INSTALL_DIR/bin/prunequest" "$BIN_DIR/prunequest"

echo "=========================================================="
echo "✔ PruneQuest Studio installed successfully!"
echo "=========================================================="
echo ""
echo "To launch PruneQuest Studio, run:"
echo "    prunequest studio"
echo ""
echo "Or using the direct environment binary:"
echo "    $INSTALL_DIR/bin/prunequest studio"
echo ""

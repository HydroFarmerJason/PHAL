#!/bin/bash
# Install system dependencies required for PHAL

set -e

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y python3-dev python3-venv build-essential libpq-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew update
    brew install python3 postgresql
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "System dependencies installed."

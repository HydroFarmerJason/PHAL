#!/bin/bash
# Install system dependencies for PHAL

set -e

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential libssl-dev libffi-dev

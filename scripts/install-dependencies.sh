#!/bin/bash
# Install system dependencies for PHAL
set -e

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip docker.io docker-compose

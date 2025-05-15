#!/bin/bash
# script for creating virtual environments for different python 3.12 version
if [ -d "venv312" ]; then
    echo "venv312 already exists, skipping creation."
else
    echo "Creating venv312..."
    # Check if python3.12 is installed
    if ! command -v python3.12 &> /dev/null
    then
        echo "python3.12 could not be found, please install it first."
        exit
    fi
    python3.12 -m venv venv312
    source venv312/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

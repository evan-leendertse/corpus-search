#!/bin/bash
# Setup script for document similarity system

echo "=== Setting up Document Similarity System ==="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Create virtual environment (optional)
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Make main.py executable
chmod +x main.py

echo "=== Setup Complete ==="
echo "To run the system:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run main.py: python main.py"
#!/bin/bash

# Check if virtual environment directory exists
if [ ! -d "venv" ]; then
  # Create a virtual environment
  python3 -m venv venv
  echo "Virtual environment created."
fi

# Activate the virtual environment
source venv/bin/activate
echo "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
echo "Required packages installed."

# Note: Do not deactivate the virtual environment
# This will leave you in the virtual environment
echo "You are now in the virtual environment. You can run your Python scripts."
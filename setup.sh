#!/bin/bash

# This script sets up a virtual environment named '.venv' one level above the current directory
# and installs the required Python packages (torch, nvcc4jupyter, numpy).

# Get the directory name where the script is being executed (the repository folder name)
REPO_DIR_NAME=$(basename "$(pwd)")

echo "Starting setup for repository: $REPO_DIR_NAME"

# --- 1. Navigate up one directory to create the .venv outside the repo ---
cd ..
if [ $? -ne 0 ]; then
    echo "Error: Could not navigate up to the parent directory. Aborting."
    exit 1
fi

echo "Creating virtual environment at: $(pwd)/.venv"

# --- 2. Create the virtual environment ---
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment using python3 -m venv. Is python3 installed? Aborting."
    exit 1
fi

# --- 3. Determine the correct activation path (handles Linux/macOS and Windows path structures) ---
if [ -f ".venv/bin/activate" ]; then
    VENV_ACTIVATE=".venv/bin/activate"
elif [ -f ".venv/Scripts/activate" ]; then
    VENV_ACTIVATE=".venv/Scripts/activate"
else
    echo "Error: Could not find venv activation script. Aborting."
    exit 1
fi

echo "Activating virtual environment..."
source "$VENV_ACTIVATE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment. Aborting."
    exit 1
fi

# --- 4. Install the required packages ---
echo "Installing packages: nvcc4jupyter, numpy..."

pip install nvcc4jupyter numpy
pip install jupyter_client ipykernel
pip install nvidia-cutlass-dsl
pip install ipywidgets
pip install bokeh
pip install sympy


if [ $? -ne 0 ]; then
    echo "Error: Failed to install one or more packages. Please check the pip output above."
    # Attempt to deactivate and return even on failure
    deactivate 2>/dev/null
    cd "$REPO_DIR_NAME"
    exit 1
fi

echo "Installation complete."

# --- 5. Deactivate and return to the repository folder ---
deactivate 2>/dev/null

echo "Returning to repository folder: $(pwd)/$REPO_DIR_NAME"
cd "$REPO_DIR_NAME"
if [ $? -ne 0 ]; then
    echo "Warning: Failed to return to the original repository directory."
fi

git config --global user.email sathya.pranav.deepak@gmail.com
git config --global user.name PranavDeepakSathya

echo "Setup finished. Run 'source ../.venv/bin/activate' (Linux/macOS) or '../.venv/Scripts/activate' (Windows) and then launch Jupyter."

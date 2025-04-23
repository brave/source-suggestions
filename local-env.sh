#!/bin/bash

# Remove existing virtual environment
rm -rf .venv

# Create a new virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Ensure the correct Python version is being used
pyenv global 3.9.11
eval "$(pyenv init --path)"

# Install the required packages
echo "Install requirements"
pip install -r requirements.txt

# Print completion messages
echo "---------------------------"
echo ".venv recreated and sourced"
echo "Set python version to 3.9.11"
echo "Installed requirements"
echo "Complete"
echo "---------------------------"

# download these files
urls=(
  "https://brave-today-cdn.brave.com/brave-today/feed.en_US.json"
  "https://brave-today-cdn.brave.com/source-suggestions/articles_history.en_US.csv"
  "https://brave-today-cdn.brave.com/sources.en_US.json"
)

for url in "${urls[@]}"; do
  # Extract filename from URL
  filename=$(basename "$url")

  # Download the file using wget
  wget -O "$filename" "$url"

  # Check if download was successful
  if [ $? -eq 0 ]; then
    echo "Successfully downloaded: $filename"
  else
    echo "Failed to download: $filename"
  fi
done


# Keep the virtual environment active
exec "$SHELL"

#!/bin/bash

echo "Downloading PlantVillage dataset..."

# Download from Kaggle
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p data/raw/

# Unzip
cd data/raw/
unzip -q new-plant-diseases-dataset.zip
cd ../..

# Verify download
echo "Dataset structure:"
ls -lh data/raw/New\ Plant\ Diseases\ Dataset\(Augmented\)/

echo "Download complete!"

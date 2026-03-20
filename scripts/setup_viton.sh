#!/bin/bash
# Quick VITON Dataset Setup Script
# Checks for existing dataset or downloads from official sources

echo "=================================================="
echo "AR Mirror - VITON Dataset Setup"
echo "=================================================="

DATASET_ROOT="dataset"

# Check current dataset status
echo -e "\n[1/4] Checking existing dataset..."
CLOTH_COUNT=$(ls ${DATASET_ROOT}/train/cloth/*.jpg 2>/dev/null | wc -l)
MASK_COUNT=$(ls ${DATASET_ROOT}/train/cloth-mask/*.jpg 2>/dev/null | wc -l)

echo "  ✓ Cloth images: $CLOTH_COUNT"
echo "  ✓ Cloth masks: $MASK_COUNT"

if [ $CLOTH_COUNT -gt 100 ]; then
    echo -e "\n✓ Full dataset detected ($CLOTH_COUNT garments)"
    exit 0
fi

echo -e "\n[2/4] Searching for VITON dataset..."
# Check common locations
SEARCH_PATHS=(
    "$HOME/Downloads"
    "$HOME/Dataset"
    "$HOME/Documents/Datasets"
    "/c/Datasets"
    "$PWD/../"
)

for path in "${SEARCH_PATHS[@]}"; do    if [ -d "$path" ]; then
        found=$(find "$path" -maxdepth 2 -type d -name "*viton*" -o -name "*VITON*" 2>/dev/null)
        if [ ! -z "$found" ]; then
            echo "  ✓ Found: $found"
        fi
    fi
done

echo -e "\n[3/4] Dataset Download Options:"
echo "---------------------------------------------------"
echo "Option 1: VITON-HD (High Quality)"
echo "  - GitHub: https://github.com/shadow2496/VITON-HD"
echo "  - Size: ~15GB (1024x768 resolution)"
echo ""
echo "Option 2: CP-VITON+ (Research)"
echo "  - GitHub: https://github.com/minar09/CP-VITON-Plus"
echo "  - Size: ~2GB (256x192 resolution)"
echo ""
echo "Option 3: DeepFashion / Zalando"
echo "  - Kaggle: Fashion Product Images Dataset"
echo "  - Alternative commercial clothing database"
echo "---------------------------------------------------"

echo -e "\n[4/4] Quick Setup Instructions:"
echo ""
echo "TO USE VITON-HD (Recommended):"
echo "  1. Download from: https://github.com/shadow2496/VITON-HD/releases"
echo "  2. Extract train.zip to dataset/train/"
echo "  3. Run: python scripts/download_viton.py --check"
echo ""
echo "TO USE EXISTING DATASET:"
echo "  1. Copy your cloth/*.jpg to dataset/train/cloth/"
echo "  2. Copy your cloth-mask/*.jpg to dataset/train/cloth-mask/"
echo "  3. Run: python app.py --phase 0 --duration 0"
echo ""
echo "=================================================="

# Utility Scripts

This directory contains utility scripts for setup, data generation, and analysis.

## Structure

- **setup/** - Installation and setup scripts
  - `download_models.py` - Download ML models
  - `download_mp_model.py` - Download MediaPipe models
  
- **generators/** - Data generation scripts
  - `generate_samples.py` - Generate garment samples
  - `generate_placeholders.py` - Generate placeholder images
  - `generate_inventory.py` - Generate inventory data
  
- **tools/** - Analysis and validation tools
  - `analyze_logs.py` - Analyze system logs
  - `validate_accuracy.py` - Validate measurement accuracy
  - `collect_validation_data.py` - Collect validation datasets
  - `validate_setup.py` - Validate system setup
  - `verify_mediapipe.py` - Verify MediaPipe installation

## Usage

### Setup Scripts

```bash
# Download all required models
python scripts/setup/download_models.py

# Download MediaPipe models specifically
python scripts/setup/download_mp_model.py
```

### Generator Scripts

```bash
# Generate sample garments
python scripts/generators/generate_samples.py

# Generate placeholder images
python scripts/generators/generate_placeholders.py

# Generate inventory data
python scripts/generators/generate_inventory.py
```

### Analysis Tools

```bash
# Analyze logs
python scripts/tools/analyze_logs.py

# Validate accuracy
python scripts/tools/validate_accuracy.py

# Collect validation data
python scripts/tools/collect_validation_data.py
```

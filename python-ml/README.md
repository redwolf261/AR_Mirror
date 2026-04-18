# Python ML Service

## Overview

This directory contains the Python machine learning service for the AR Mirror system, including pose detection, body measurement, and size recommendation.

## Quick Start

```bash
# Install dependencies (standalone python-ml usage)
pip install -r requirements.txt

# If you are running the full AR Mirror app from repo root,
# install root dependencies first to keep one consistent environment:
# pip install -r ../requirements.txt

# Download models
python ../scripts/setup/download_models.py

# Run demo
python tests/demos/demo.py

# Run API server
python src/orchestrator.py
```

## Structure

- `src/` - Core modules and pipelines
- `tests/` - Unit tests, integration tests, and demos
- `data/` - Sample data and databases
- `models/` - ML models (gitignored, download via scripts)

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## API Usage

The ML service can be run as a standalone FastAPI server:

```python
# Start server
python src/python_ml_service.py

# API will be available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

See `src/orchestrator.py` for full system integration.

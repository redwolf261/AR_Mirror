# Changelog

All notable changes to the AR Mirror project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.0.0] - 2026-02-01

### Major Restructuring 🎉

This release represents a complete project reorganization for better maintainability and developer experience.

### Added
- ✅ Organized directory structure (python-ml/, backend/, docs/, scripts/, config/)
- ✅ Comprehensive `requirements.txt` with all 14 Python dependencies
- ✅ Developer requirements file (`requirements-dev.txt`) with 11 dev tools
- ✅ README files for all major subdirectories
- ✅ Complete `.gitignore` with proper exclusion rules
- ✅ New comprehensive main README.md with quick start guide
- ✅ Python package structure with `__init__.py`

### Changed
- 📁 Moved 54+ files from root to organized subdirectories
  - Core ML modules → `python-ml/src/`
  - Test files → `python-ml/tests/`
  - Utility scripts → `scripts/`
  - Data files → `python-ml/data/`
  - Model files → `python-ml/models/`
  - Configuration → `config/`
  - Documentation → `docs/`

- 📝 Consolidated documentation from 26 files to 8 essential files (78% reduction)
  - Moved to `docs/` directory
  - Renamed for clarity (e.g., DATA_LOGGING_SCHEMA.md → data-strategy.md)

### Removed
- ❌ Deleted 21+ unnecessary/redundant files:
  - `get-pip.py` (2MB) - standard Python installer
  - `python_env.zip` (8.6MB) - committed environment
  - `CODE_FROZEN.txt` - status file
  - 18 redundant documentation files
  
### Fixed
- 🐛 Fixed incomplete `requirements.txt` (was only 3 deps, now 14)
- 🐛 Updated `.gitignore` to prevent committing large binaries
- 🐛 Organized scattered test files into proper structure

### Migration Guide

If you have an existing clone:

```bash
# Backup your work
git stash

# Pull latest changes
git pull

# Reinstall dependencies
cd python-ml
pip install -r requirements.txt

# Download models to new location
python ../scripts/setup/download_models.py

# Run tests to verify
pytest tests/
```

### Breaking Changes

⚠️ File paths have changed. Update any scripts/imports:

- `sizing_pipeline.py` → `python-ml/src/sizing_pipeline.py`
- `test_*.py` → `python-ml/tests/unit/` or `python-ml/tests/integration/`
- `demo.py` → `python-ml/tests/demos/demo.py`
- Documentation files → `docs/`

---

## [1.0.0] - 2026-01-XX

### Initial Release

- Initial AR Mirror implementation
- MediaPipe-based pose detection
- Body measurement estimation
- Size recommendation system
- Garment visualization
- Backend API (NestJS)
- Mobile support planning

# Free SMPL Setup Guide

## 🆓 Quick Start with Free Models

This guide gets you started with **free SMPL-X models** for development and prototyping before upgrading to commercial licensing.

## 📥 Free SMPL-X Model Setup

### Option 1: Hugging Face (Recommended)

```bash
# Install dependencies
pip install huggingface_hub

# Download SMPL-X neutral model
python -c "
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path

# Create models directory
Path('smpl_models').mkdir(exist_ok=True)

# Download SMPL-X neutral model
try:
    model_path = hf_hub_download(
        repo_id='vchoutas/smplx', 
        filename='SMPLX_NEUTRAL.npz'
    )
    shutil.copy(model_path, 'smpl_models/SMPLX_NEUTRAL.npz')
    print('✅ Free SMPL-X model downloaded successfully')
except Exception as e:
    print(f'⚠️  Download failed: {e}')
    print('💡 Creating development placeholder...')
    
    import numpy as np
    placeholder = {
        'vertices': np.zeros((6890, 3), dtype=np.float32),
        'faces': np.random.randint(0, 6890, (13776, 3)),
        'pose_params': np.zeros(75),
        'shape_params': np.zeros(10)
    }
    np.savez('smpl_models/SMPLX_NEUTRAL.npz', **placeholder)
    print('🔧 Development placeholder created')
"
```

### Option 2: Manual Download

1. Visit [SMPL-X GitHub Repository](https://github.com/vchoutas/smplx)
2. Download `SMPLX_NEUTRAL.npz` 
3. Place in `smpl_models/SMPLX_NEUTRAL.npz`

### Option 3: Development Placeholder

```bash
# Create minimal development placeholder
python -c "
import numpy as np
from pathlib import Path

Path('smpl_models').mkdir(exist_ok=True)

# Minimal SMPL structure for development
placeholder = {
    'vertices': np.zeros((6890, 3), dtype=np.float32),     # Standard SMPL vertex count
    'faces': np.random.randint(0, 6890, (13776, 3)),      # Standard SMPL face count  
    'pose_params': np.zeros(75),                          # 25 joints × 3 rotations
    'shape_params': np.zeros(10),                         # 10 beta parameters
    'joint_regressor': np.random.rand(24, 6890)          # Joint to vertex mapping
}

np.savez('smpl_models/SMPLX_NEUTRAL.npz', **placeholder)
print('✅ Development placeholder created')
print('💡 This allows testing without real SMPL data')
print('🔄 Upgrade to real models for accurate results')
"
```

## 🚀 Test Free Setup

### Basic Factory Test

```bash
# Test synthetic data factory with free models
python -c "
from scripts.synthetic_data_factory import SyntheticDataFactory, GenerationConfig
from pathlib import Path

config = GenerationConfig(
    output_dir=Path('test_output'),
    batch_size=5,
    resolution=(512, 512)
)

# Initialize with free SMPL (default)
factory = SyntheticDataFactory(config, use_commercial_smpl=False)

print('✅ Free SMPL factory initialized successfully')
print('🔧 Ready for development and prototyping')
"
```

### Generate Test Samples

```bash
# Generate sample batch with free models
python scripts/production_pipeline.py \
  --phase 1 \
  --batch_size 10 \
  --synthetic_only \
  --output_dir test_free_samples
```

## 📊 Free vs. Commercial Comparison

| Feature | Free SMPL-X | Commercial SMPL |
|---------|-------------|-----------------|
| **Cost** | €0 | €100-€500/year |
| **License** | Academic research | Commercial use |
| **Deployment** | Development only | Production ready |
| **Model Quality** | Basic neutral | Male/Female variants |
| **Support** | Community | Official MPI support |
| **Legal Status** | Non-commercial | Full commercial rights |

## 🔄 Upgrade Path to Commercial

### When to Upgrade

✅ **Stay with Free** for:
- Initial prototyping
- Algorithm development  
- Proof-of-concept validation
- Academic research

✅ **Upgrade to Commercial** for:
- Production deployment
- Commercial applications
- Customer-facing products
- Revenue-generating use

### Migration Process

1. **Acquire License**: Contact MPI-IS (smpl@tue.mpg.de)
2. **Install Models**: Place commercial models in `smpl_models/`
3. **Update Code**: Set `use_commercial_smpl=True`
4. **Validate**: Test production pipeline

```bash
# Before: Free development
factory = SyntheticDataFactory(config, use_commercial_smpl=False)

# After: Commercial production
factory = SyntheticDataFactory(config, use_commercial_smpl=True)
```

## ⚠️ License Compliance

### Free SMPL-X Restrictions
- ❌ **No commercial use** - Development and research only
- ❌ **No redistribution** - Cannot include in commercial products
- ❌ **No warranty** - Use at own risk for prototyping

### Safe Development Practices
- ✅ Use free models for development and testing
- ✅ Switch to commercial before production deployment  
- ✅ Document upgrade timeline in project planning
- ✅ Validate results with placeholder data first

## 🛠️ Technical Notes

### Model Compatibility
```python
# Free SMPL-X format (.npz)
data = np.load('smpl_models/SMPLX_NEUTRAL.npz')
vertices = data['vertices']  # Direct numpy access

# Commercial SMPL format (.pkl) 
with open('smpl_models/SMPL_NEUTRAL.pkl', 'rb') as f:
    model = pickle.load(f)  # Requires specific SMPL library
```

### Performance Considerations
- **Free models**: Limited shape/pose variation
- **Commercial models**: Full body shape diversity
- **Placeholder data**: Fast testing, no meaningful output
- **Real models**: Accurate geometry, slower processing

## 📞 Support

- **Free SMPL-X Issues**: [GitHub Issues](https://github.com/vchoutas/smplx/issues)
- **Commercial Licensing**: smpl@tue.mpg.de
- **Development Questions**: Use placeholder data for testing

---

## 🎯 Quick Commands

```bash
# Setup free environment
pip install huggingface_hub numpy
python scripts/setup_free_smpl.py

# Test with free models
python scripts/production_pipeline.py --batch_size 10 --synthetic_only

# Upgrade to commercial (after license acquisition)
python scripts/production_pipeline.py --batch_size 1000 --use_blender --commercial
```

**Status**: 🆓 Free development ready, commercial upgrade path defined
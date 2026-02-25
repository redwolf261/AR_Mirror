import pytest
import sys
import os
from pathlib import Path

# Add src to python path to allow imports from src.
# This logic ensures that when pytest runs from python-ml/tests, it can find src.
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Also add the python-ml directory itself to path so 'src' module can be found if imported as 'src.xxx'
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

@pytest.fixture
def data_dir():
    return Path(__file__).parent.parent / "data"

@pytest.fixture
def model_dir():
    return Path(__file__).parent.parent / "models"

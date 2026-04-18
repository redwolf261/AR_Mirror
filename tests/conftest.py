from pathlib import Path
import sys

# Ensure repo-root imports (for example app and src.*) work from any tests/* module.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

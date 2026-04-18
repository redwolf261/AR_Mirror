#!/usr/bin/env python3
"""
Convert SMPL v1.1.0 pkl files (which contain chumpy arrays) to pure-numpy pkl files.

Uses a custom Unpickler that intercepts chumpy class references and replaces them
with plain Python stubs, then converts everything to numpy and re-saves.
No chumpy installation required.

Usage:
    python scripts/fix_smpl_pkl.py
"""
import sys
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Custom Unpickler — redirects chumpy types to a plain stub
# ---------------------------------------------------------------------------

class _ChStub:
    """Absorbs any chumpy object state without requiring chumpy installed."""
    pass


class _ChumfreeUnpickler(pickle.Unpickler):
    def __init__(self, f):
        super().__init__(f, encoding="latin1", errors="replace")

    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChStub
        if module in ("scipy.sparse.csc", "scipy.sparse.csr", "scipy.sparse"):
            try:
                import scipy.sparse
                cls = getattr(scipy.sparse, name, None)
                if cls is not None:
                    return cls
            except ImportError:
                pass
        return super().find_class(module, name)


def _to_numpy(obj):
    """Recursively convert _ChStub instances and numpy subclasses to plain ndarrays."""
    if isinstance(obj, _ChStub):
        # Try common chumpy internal attribute names that hold the array data
        for attr in ('r', 'x', '_vals', 'val', 'data', 'dr_lookup'):
            v = obj.__dict__.get(attr)
            if isinstance(v, np.ndarray):
                return v
        # Fallback: any ndarray value in the stub's dict
        for v in obj.__dict__.values():
            if isinstance(v, np.ndarray):
                return v
        return np.zeros(1, dtype=np.float32)

    if isinstance(obj, np.ndarray):
        return np.asarray(obj)  # strip any ndarray subclass

    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        converted = [_to_numpy(v) for v in obj]
        return type(obj)(converted)

    try:
        import scipy.sparse as sp
        if sp.issparse(obj):
            return obj  # keep sparse; smpl_body_reconstruction calls .toarray()
    except ImportError:
        pass

    return obj


def fix_pkl(src: Path, dst: Path) -> bool:
    print(f"  Converting: {src.name}")
    try:
        with open(src, "rb") as f:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = _ChumfreeUnpickler(f).load()
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return False

    data_clean = _to_numpy(data)

    try:
        with open(dst, "wb") as f:
            pickle.dump(data_clean, f, protocol=2)
    except Exception as e:
        print(f"  ERROR saving: {e}")
        return False

    src_mb = src.stat().st_size / 1_048_576
    dst_mb = dst.stat().st_size / 1_048_576
    print(f"  OK  {src_mb:.1f}MB → {dst_mb:.1f}MB")
    return True


def main():
    targets = ["smpl_neutral.pkl", "smpl_male.pkl", "smpl_female.pkl"]

    print("Converting SMPL pkl files to pure-numpy (removing chumpy dependency)...")
    ok = 0
    for name in targets:
        src = MODELS_DIR / name
        if not src.exists():
            print(f"  SKIP (not found): {name}")
            continue
        tmp = MODELS_DIR / (name + ".tmp")
        if fix_pkl(src, tmp):
            tmp.replace(src)
            ok += 1
        else:
            tmp.unlink(missing_ok=True)

    print(f"\nDone: {ok}/{len(targets)} files converted.")
    if ok == len(targets):
        print("All SMPL pkl files are now chumpy-free. ✓")
    else:
        print("Some files failed — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
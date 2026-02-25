"""
Size chart mapper and brand calibration.

SizeChart maps body measurements (heuristic ratios or predicted β) to
collar / jacket / trouser size strings, with a confidence level for each.

Phase 1: heuristic threshold tables from generic.json.
Phase 2: argmax over classifier logits.
Phase 3: calibrate() fits per-brand LogisticRegression on historical orders.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

_CHARTS_DIR = Path(__file__).parent / "size_charts"

# Confidence margin thresholds for High / Medium / Low
_HIGH_MARGIN   = 0.40
_MEDIUM_MARGIN = 0.20


def _load_chart(brand: str) -> dict:
    path = _CHARTS_DIR / f"{brand}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Size chart not found: {path}. "
            f"Available: {[p.stem for p in _CHARTS_DIR.glob('*.json')]}"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _heuristic_class(value: float, breakpoints: list[float], classes: list[int]) -> int:
    """Return class index using threshold breakpoints."""
    for i, bp in enumerate(breakpoints):
        if value <= bp:
            return classes[i]
    return classes[-1]


def _margin_to_confidence(sorted_probs: np.ndarray) -> str:
    """Convert top-2 probability margin to High / Medium / Low label."""
    if len(sorted_probs) < 2:
        return "High"
    margin = float(sorted_probs[-1] - sorted_probs[-2])
    if margin >= _HIGH_MARGIN:
        return "High"
    if margin >= _MEDIUM_MARGIN:
        return "Medium"
    return "Low"


class SizeChart:
    """
    Men's formalwear size chart.

    Args:
        brand : chart name matching a file in size_charts/{brand}.json.
    """

    def __init__(self, brand: str = "generic") -> None:
        self.brand = brand
        self._data = _load_chart(brand)
        self._calibrated_models: dict[str, Any] = {}  # sklearn models per head

        # Load calibrated models if saved in chart json
        for head in ("collar", "jacket", "trouser"):
            raw = self._data.get("calibration", {}).get(f"{head}_model")
            if raw and isinstance(raw, str):
                try:
                    import base64
                    self._calibrated_models[head] = pickle.loads(
                        base64.b64decode(raw.encode())
                    )
                except Exception:
                    pass

    # ── Public API ────────────────────────────────────────────────────────

    def predict_size(
        self,
        measurements_or_beta,     # BodyProxyMeasurements | np.ndarray[10] | torch.Tensor[B,10]
        logits: Optional[dict] = None,
    ) -> dict:
        """
        Predict collar, jacket, and trouser size.

        Phase 1 (logits=None):
            Uses heuristic threshold tables from the chart JSON.
            measurements_or_beta must be a BodyProxyMeasurements instance.

        Phase 2+ (logits provided):
            logits = {"collar": np.ndarray[8], "jacket": np.ndarray[8], "trouser": np.ndarray[10]}
            Uses argmax over logits.  Calibrated sklearn heads used if available.

        Returns:
            {
                "collar":          "16.0",
                "jacket":          "42R",
                "trouser_waist":   "34",
                "confidence_level": "High" | "Medium" | "Low" | "heuristic",
            }
        """
        if logits is not None:
            return self._predict_from_logits(logits)
        return self._predict_heuristic(measurements_or_beta)

    def calibrate(
        self,
        pilot_log_path: str | Path,
        purchase_data: list[dict],
    ) -> None:
        """
        Fit a LogisticRegression per head on brand purchase history.

        No GPU required.  Runs in minutes.

        Args:
            pilot_log_path : path to pilot_log.jsonl.
            purchase_data  : list of dicts with keys:
                               session_id   : str
                               size_purchased : str (e.g. "42R", "16.0", "34")
                               returned     : bool
        """
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
        except ImportError as e:
            raise ImportError("scikit-learn required for calibration. pip install scikit-learn") from e

        import base64

        sessions = self._load_pilot_sessions(pilot_log_path)
        purchase_map = {d["session_id"]: d for d in purchase_data}

        # Build feature matrix from logged β or ratios
        # For Phase 1 logs (no β), we use the ratio features
        features_by_head: dict[str, list] = {"collar": [], "jacket": [], "trouser": []}
        labels_by_head:   dict[str, list] = {"collar": [], "jacket": [], "trouser": []}

        for sess in sessions:
            sid = sess.get("session_id")
            if sid not in purchase_map:
                continue
            pd = purchase_map[sid]
            if pd.get("returned"):
                continue  # only train on non-returns (successful fits)

            # Feature: ratio vector [shoulder_w, chest_w, hip_w, torso_depth, height_norm]
            feats = self._session_to_features(sess)
            if feats is None:
                continue

            size_str = pd["size_purchased"]
            for head in ("collar", "jacket", "trouser"):
                cls_idx = self._size_to_class_idx(head, size_str)
                if cls_idx is not None:
                    features_by_head[head].append(feats)
                    labels_by_head[head].append(cls_idx)

        chart_updated = False
        for head in ("collar", "jacket", "trouser"):
            X = np.array(features_by_head[head])
            y = np.array(labels_by_head[head])
            if len(set(y)) < 2 or len(X) < 10:
                logger.warning("Calibration: not enough data for '%s' head (%d samples).", head, len(X))
                continue

            clf = LogisticRegression(max_iter=500, multi_class="multinomial")
            clf.fit(X, y)
            self._calibrated_models[head] = clf

            # Serialise into chart JSON
            model_bytes = pickle.dumps(clf)
            self._data.setdefault("calibration", {})[f"{head}_model"] = (
                base64.b64encode(model_bytes).decode()
            )
            logger.info("Calibrated '%s' head on %d samples.", head, len(X))
            chart_updated = True

        if chart_updated:
            self._data.setdefault("calibration", {})["calibrated_at"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._save_chart()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _predict_heuristic(self, measurements) -> dict:
        thresh = self._data.get("heuristic_thresholds", {})

        # collar ← shoulder_width_ratio
        c_thresh = thresh.get("collar", {})
        collar_idx = _heuristic_class(
            measurements.shoulder_width_ratio,
            c_thresh.get("shoulder_width_ratio_breakpoints", []),
            c_thresh.get("class_at_or_below", list(range(8))),
        )

        # jacket ← chest_width_ratio
        j_thresh = thresh.get("jacket", {})
        jacket_idx = _heuristic_class(
            measurements.chest_width_ratio,
            j_thresh.get("chest_width_ratio_breakpoints", []),
            j_thresh.get("class_at_or_below", list(range(8))),
        )

        # trouser ← hip_width_ratio
        t_thresh = thresh.get("trouser", {})
        trouser_idx = _heuristic_class(
            measurements.hip_width_ratio,
            t_thresh.get("hip_width_ratio_breakpoints", []),
            t_thresh.get("class_at_or_below", list(range(10))),
        )

        collar  = self._class_to_size("collar",  collar_idx)
        jacket  = self._class_to_size("jacket",  jacket_idx)
        trouser = self._class_to_size("trouser", trouser_idx)

        return {
            "collar":           collar,
            "jacket":           jacket,
            "trouser_waist":    trouser,
            "confidence_level": "heuristic",
        }

    def _predict_from_logits(self, logits: dict) -> dict:
        results = {}
        confs   = []
        for head, key, size_key in [
            ("collar",  "collar",  "collar"),
            ("jacket",  "jacket",  "jacket"),
            ("trouser", "trouser", "trouser_waist"),
        ]:
            raw = np.array(logits[head], dtype=np.float32)

            if head in self._calibrated_models:
                # Use brand-calibrated model
                clf = self._calibrated_models[head]
                probs = clf.predict_proba(raw.reshape(1, -1))[0]
            else:
                probs = _softmax(raw)

            cls_idx = int(np.argmax(probs))
            results[size_key] = self._class_to_size(head, cls_idx)
            confs.append(_margin_to_confidence(np.sort(probs)))

        # Overall confidence: take the minimum (most conservative)
        conf_rank = {"High": 2, "Medium": 1, "Low": 0}
        overall = min(confs, key=lambda c: conf_rank[c])
        results["confidence_level"] = overall
        return results

    def _class_to_size(self, head: str, class_idx: int) -> str:
        key_map = {
            "collar":  ("collar_sizes",  "classes"),
            "jacket":  ("jacket_sizes",  "classes"),
            "trouser": ("trouser_sizes", "classes"),
        }
        section, arr_key = key_map[head]
        classes = self._data[section][arr_key]
        class_idx = max(0, min(class_idx, len(classes) - 1))
        val = classes[class_idx]
        if head == "collar":
            return str(float(val))
        return str(int(val))

    def _size_to_class_idx(self, head: str, size_str: str) -> Optional[int]:
        """Convert a size string like '42R' or '16.0' to class index."""
        key_map = {
            "collar":  "collar_sizes",
            "jacket":  "jacket_sizes",
            "trouser": "trouser_sizes",
        }
        section = self._data.get(key_map[head], {})
        classes = section.get("classes", [])
        # Strip length suffix for jacket (e.g. "42R" → 42)
        clean = size_str.replace("R", "").replace("S", "").replace("L", "").strip()
        try:
            numeric = float(clean)
        except ValueError:
            return None
        for i, c in enumerate(classes):
            if abs(float(c) - numeric) < 0.01:
                return i
        return None

    def _session_to_features(self, sess: dict) -> Optional[np.ndarray]:
        pred = sess.get("predicted_size")
        if pred is None:
            return None
        # Use stored ratio proxies if available (Phase 2+), else fallback
        ratios = sess.get("measurement_ratios")
        if ratios:
            return np.array([
                ratios.get("shoulder_width_ratio", 0.0),
                ratios.get("chest_width_ratio",    0.0),
                ratios.get("hip_width_ratio",      0.0),
                ratios.get("torso_depth_ratio",    0.0),
                ratios.get("height_norm",          0.0),
            ], dtype=np.float32)
        return None

    def _load_pilot_sessions(self, path: str | Path) -> list[dict]:
        import jsonlines  # type: ignore  (graceful import)
        sessions = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sessions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return sessions

    def _save_chart(self) -> None:
        path = _CHARTS_DIR / f"{self.brand}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        logger.info("Updated chart saved → %s", path)

    def lookup_collar(self, beta_or_ratios) -> int:
        """Return collar class index from beta or measurement ratios (Phase 2)."""
        result = self.predict_size(beta_or_ratios)
        return self._data["collar_sizes"]["class_index"].get(result["collar"], 2)

    def lookup_jacket(self, beta_or_ratios) -> int:
        """Return jacket class index."""
        result = self.predict_size(beta_or_ratios)
        return self._data["jacket_sizes"]["class_index"].get(result["jacket"], 3)

    def lookup_trouser(self, beta_or_ratios) -> int:
        """Return trouser class index."""
        result = self.predict_size(beta_or_ratios)
        return self._data["trouser_sizes"]["class_index"].get(result["trouser_waist"], 3)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

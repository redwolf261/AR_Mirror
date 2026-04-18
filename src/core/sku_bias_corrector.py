"""
SKU Bias Corrector — Phase 4 Per-SKU Auto-Learning

Reads user_feedback.jsonl (written by SessionLogger) and automatically
computes per-SKU measurement bias corrections.

How the moat works:
  Session → Purchase → Return → "too tight in chest" → +2cm bias for SKU chest

After enough samples per SKU:
  - Shoulder, chest, hip, length tolerances are auto-tuned
  - Indian brand-specific sizing quirks are captured
  - Corrections compound over time — competitors need the same dataset to match

Corrections are persisted to learned_corrections/sku_corrections.json and
applied at prediction time (zero extra latency — just a dict lookup).

Usage:
    corrector = SKUBiasCorrector()
    corrector.update_from_feedback_log()   # re-learns from full feedback log
    delta = corrector.get_correction("TSH-001")
    adjusted_chest = measured_chest_cm + delta.chest_bias_cm
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

_CORRECTIONS_FILE = Path(__file__).parent.parent.parent / "learned_corrections" / "sku_corrections.json"
_FEEDBACK_LOG = Path(__file__).parent.parent.parent / "logs" / "user_feedback.jsonl"

# Minimum feedback samples before a correction is considered trustworthy
_MIN_SAMPLES = 10
# Correction decays after 90 days (seasonal sizing shifts)
_TTL_DAYS = 90


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SKUCorrection:
    sku: str
    # Signed bias to ADD to raw measurement before size lookup (cm)
    shoulder_bias_cm: float = 0.0
    chest_bias_cm: float = 0.0
    hip_bias_cm: float = 0.0
    length_bias_cm: float = 0.0
    # Legacy tolerance fields (kept for back-compat with v1.0 schema)
    shoulder_tolerance_cm: float = 0.0
    chest_tolerance_cm: float = 0.0
    hip_tolerance_cm: Optional[float] = None
    inseam_tolerance_cm: Optional[float] = None
    # Meta
    confidence: float = 0.0          # 0-1, based on sample count
    samples_used: int = 0
    accuracy: float = 0.0            # fraction of non-returned purchases
    created_at: str = ""
    updated_at: str = ""
    expires_at: str = ""


# ---------------------------------------------------------------------------
# Corrector
# ---------------------------------------------------------------------------

class SKUBiasCorrector:
    """
    Learns and applies per-SKU measurement bias corrections from return data.

    The core insight: if a garment with size_label="M" (chest=50cm) is
    consistently returned because it's "too tight" by people whose measured
    chest is 47cm, then the true cutoff for that SKU's "M" is actually 48cm.
    We learn that -2cm bias automatically.

    Backed by `learned_corrections/sku_corrections.json` (auto-created).
    """

    def __init__(
        self,
        corrections_file: Optional[Path] = None,
        feedback_log: Optional[Path] = None,
    ):
        self.corrections_file = corrections_file or _CORRECTIONS_FILE
        self.feedback_log = feedback_log or _FEEDBACK_LOG
        self.corrections_file.parent.mkdir(parents=True, exist_ok=True)

        self._corrections: Dict[str, SKUCorrection] = {}
        self._load()

    # ------------------------------------------------------------------
    # Apply corrections
    # ------------------------------------------------------------------

    def get_correction(self, sku: str) -> SKUCorrection:
        """
        Return the bias correction for a SKU.
        Returns a zero-bias correction if the SKU is unknown or has <10 samples.
        """
        corr = self._corrections.get(sku)
        if corr is None or corr.samples_used < _MIN_SAMPLES:
            return SKUCorrection(sku=sku)
        if self._is_expired(corr):
            logger.debug(f"Correction for {sku} has expired — returning zero bias")
            return SKUCorrection(sku=sku)
        return corr

    def apply(
        self,
        sku: str,
        shoulder_cm: float,
        chest_cm: float,
        hip_cm: float = 0.0,
        length_cm: float = 0.0,
    ) -> Dict[str, float]:
        """
        Apply bias corrections to raw body measurements.

        Returns corrected measurements dict ready for size lookup.
        """
        c = self.get_correction(sku)
        return {
            "shoulder_cm": shoulder_cm + c.shoulder_bias_cm,
            "chest_cm": chest_cm + c.chest_bias_cm,
            "hip_cm": hip_cm + c.hip_bias_cm,
            "length_cm": length_cm + c.length_bias_cm,
        }

    # ------------------------------------------------------------------
    # Learn from feedback log
    # ------------------------------------------------------------------

    def update_from_feedback_log(self) -> int:
        """
        Re-scan user_feedback.jsonl and recompute all SKU corrections.

        Returns number of SKUs updated.
        """
        if not self.feedback_log.exists():
            logger.info("No feedback log found — nothing to learn from yet")
            return 0

        # Collect per-SKU feedback buckets
        # bucket: sku → list of {fit_comment, returned, body_measurements, predicted_fit}
        buckets: Dict[str, List[dict]] = defaultdict(list)

        try:
            with open(self.feedback_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        sku = rec.get("sku")
                        if not sku:
                            continue
                        buckets[sku].append(rec)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to read feedback log: {e}")
            return 0

        updated = 0
        for sku, records in buckets.items():
            if len(records) < _MIN_SAMPLES:
                continue
            corr = self._compute_correction(sku, records)
            self._corrections[sku] = corr
            updated += 1

        if updated:
            self._save()
            logger.info(f"✓ SKU corrections updated: {updated} SKUs from {sum(len(v) for v in buckets.values())} feedback records")

        return updated

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _compute_correction(self, sku: str, records: List[dict]) -> SKUCorrection:
        """
        Derive bias corrections from return comments and body measurement deltas.

        Strategy:
          1. Find all "too tight" returns → the user's chest was LARGER than predicted OK
             → bias = mean(user.chest - garment.chest_spec) for these cases
          2. Find all "too loose" returns → inverse
          3. Net signed bias = tight_bias - loose_bias
        """
        shoulder_deltas: List[float] = []
        chest_deltas: List[float] = []
        hip_deltas: List[float] = []
        length_deltas: List[float] = []
        purchased_count = 0
        returned_count = 0

        for rec in records:
            outcome = rec.get("actual_outcome", {})
            if not outcome.get("purchased"):
                continue
            purchased_count += 1

            body_meas = rec.get("body_measurements", {})
            comment = (outcome.get("fit_comment") or "").lower()
            returned = outcome.get("returned", False)

            if returned:
                returned_count += 1
                # Decode direction from fit comment
                direction = self._comment_to_direction(comment)

                # For each measurement, compute the signed delta
                # "too tight" → user is bigger than garment allows → positive delta (add cm to be safe)
                # "too loose" → user is smaller → negative delta
                sc = body_meas.get("shoulder_width_cm", 0)
                cc = body_meas.get("chest_width_cm", 0)
                hc = body_meas.get("hip_width_cm", 0)
                lc = body_meas.get("torso_length_cm", 0)

                if sc > 0:
                    shoulder_deltas.append(direction * sc * 0.05)  # 5% nudge
                if cc > 0:
                    chest_deltas.append(direction * cc * 0.05)
                if hc > 0:
                    hip_deltas.append(direction * hc * 0.05)
                if lc > 0:
                    length_deltas.append(direction * lc * 0.03)

        def safe_mean(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        accuracy = (purchased_count - returned_count) / purchased_count if purchased_count else 0.0
        confidence = min(1.0, len(records) / 100.0)   # saturates at 100 samples

        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        expires = time.strftime(
            "%Y-%m-%dT%H:%M:%S",
            time.localtime(time.time() + _TTL_DAYS * 86400),
        )

        return SKUCorrection(
            sku=sku,
            shoulder_bias_cm=round(safe_mean(shoulder_deltas), 3),
            chest_bias_cm=round(safe_mean(chest_deltas), 3),
            hip_bias_cm=round(safe_mean(hip_deltas), 3),
            length_bias_cm=round(safe_mean(length_deltas), 3),
            # Legacy fields (backward compat)
            shoulder_tolerance_cm=abs(safe_mean(shoulder_deltas)),
            chest_tolerance_cm=abs(safe_mean(chest_deltas)),
            confidence=round(confidence, 3),
            samples_used=len(records),
            accuracy=round(accuracy, 3),
            created_at=self._corrections.get(sku, SKUCorrection(sku=sku)).created_at or now,
            updated_at=now,
            expires_at=expires,
        )

    @staticmethod
    def _comment_to_direction(comment: str) -> float:
        """
        Map fit comment text to a signed direction (+1 = too tight, -1 = too loose).
        """
        tight_keywords = {"tight", "small", "snug", "narrow", "short"}
        loose_keywords = {"loose", "large", "baggy", "wide", "long", "big"}

        for kw in tight_keywords:
            if kw in comment:
                return +1.0
        for kw in loose_keywords:
            if kw in comment:
                return -1.0
        return 0.0   # unknown — discard this record's contribution

    def _is_expired(self, corr: SKUCorrection) -> bool:
        if not corr.expires_at:
            return False
        try:
            import datetime
            exp = datetime.datetime.fromisoformat(corr.expires_at)
            return datetime.datetime.now() > exp
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.corrections_file.exists():
            logger.debug("No existing corrections file — starting fresh")
            return
        try:
            with open(self.corrections_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for sku, data in raw.get("corrections", {}).items():
                # Merge old v1.0 schema (tolerance_cm) with new bias_cm fields
                self._corrections[sku] = SKUCorrection(
                    sku=data.get("sku", sku),
                    shoulder_bias_cm=data.get("shoulder_bias_cm", 0.0),
                    chest_bias_cm=data.get("chest_bias_cm", 0.0),
                    hip_bias_cm=data.get("hip_bias_cm", 0.0),
                    length_bias_cm=data.get("length_bias_cm", 0.0),
                    shoulder_tolerance_cm=data.get("shoulder_tolerance_cm", 0.0),
                    chest_tolerance_cm=data.get("chest_tolerance_cm", 0.0),
                    hip_tolerance_cm=data.get("hip_tolerance_cm"),
                    inseam_tolerance_cm=data.get("inseam_tolerance_cm"),
                    confidence=data.get("confidence", data.get("confidence_multiplier", 0.0)),
                    samples_used=data.get("samples_used", 0),
                    accuracy=data.get("accuracy", 0.0),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    expires_at=data.get("expires_at", ""),
                )
            logger.info(f"✓ Loaded {len(self._corrections)} SKU corrections")
        except Exception as e:
            logger.warning(f"Failed to load corrections: {e}")

    def _save(self) -> None:
        payload = {
            "version": "2.0",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_skus": len(self._corrections),
            "corrections": {sku: asdict(c) for sku, c in self._corrections.items()},
        }
        tmp = self.corrections_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self.corrections_file)   # atomic write
        logger.debug(f"Corrections saved: {len(self._corrections)} SKUs")

    def summary(self) -> Dict[str, int]:
        """Return a brief summary of the corrections database."""
        trusted = sum(1 for c in self._corrections.values() if c.samples_used >= _MIN_SAMPLES)
        return {
            "total_skus": len(self._corrections),
            "trusted_skus": trusted,
            "min_samples_threshold": _MIN_SAMPLES,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_default_corrector: Optional[SKUBiasCorrector] = None


def get_sku_corrector() -> SKUBiasCorrector:
    global _default_corrector
    if _default_corrector is None:
        _default_corrector = SKUBiasCorrector()
    return _default_corrector


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    corrector = SKUBiasCorrector()
    n = corrector.update_from_feedback_log()
    print(f"SKUs updated from feedback: {n}")
    print(f"Summary: {corrector.summary()}")

    # Apply correction
    result = corrector.apply("TSH-001", shoulder_cm=42.0, chest_cm=46.0, hip_cm=0.0, length_cm=63.0)
    print(f"Corrected measurements: {result}")

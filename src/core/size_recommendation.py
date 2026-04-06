"""AR Mirror decision layer: size scoring + fit classification."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Approximate retail size chart (cm). Chest/waist are circumferences.
SIZE_CHART = {
    "XS": {
        "chest_min": 81, "chest_max": 86,
        "waist_min": 67, "waist_max": 73,
        "shoulder_min": 38, "shoulder_max": 41,
        "torso_length_min": 60, "torso_length_max": 65,
    },
    "S": {
        "chest_min": 86, "chest_max": 91,
        "waist_min": 73, "waist_max": 79,
        "shoulder_min": 41, "shoulder_max": 44,
        "torso_length_min": 65, "torso_length_max": 68,
    },
    "M": {
        "chest_min": 91, "chest_max": 97,
        "waist_min": 79, "waist_max": 85,
        "shoulder_min": 44, "shoulder_max": 47,
        "torso_length_min": 68, "torso_length_max": 71,
    },
    "L": {
        "chest_min": 97, "chest_max": 102,
        "waist_min": 85, "waist_max": 91,
        "shoulder_min": 47, "shoulder_max": 50,
        "torso_length_min": 71, "torso_length_max": 74,
    },
    "XL": {
        "chest_min": 102, "chest_max": 107,
        "waist_min": 91, "waist_max": 97,
        "shoulder_min": 50, "shoulder_max": 53,
        "torso_length_min": 74, "torso_length_max": 77,
    },
    "XXL": {
        "chest_min": 107, "chest_max": 112,
        "waist_min": 97, "waist_max": 103,
        "shoulder_min": 53, "shoulder_max": 56,
        "torso_length_min": 77, "torso_length_max": 80,
    },
}

# Pixel-to-cm conversion factors (calibrated for typical webcam at 1.5-2m distance)
# These are rough estimates - would need proper camera calibration for accuracy
PIXEL_TO_CM_FACTORS = {
    "shoulder_width": 0.35,  # ~45cm shoulder = 128px @ 1.5m distance
    "chest_width": 0.38,     # Similar but slightly wider measurement
    "torso_length": 0.42,    # ~70cm torso = 167px @ 1.5m distance
}

def pixels_to_cm(pixel_measurement: float, measurement_type: str) -> float:
    """
    Convert pixel measurements to centimeters.

    Args:
        pixel_measurement: Measurement in pixels
        measurement_type: One of 'shoulder_width', 'chest_width', 'torso_length'

    Returns:
        Measurement in centimeters
    """
    factor = PIXEL_TO_CM_FACTORS.get(measurement_type, 0.4)
    return pixel_measurement * factor

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _range_mid(mn: float, mx: float) -> float:
    return (mn + mx) * 0.5


def _norm_error(v: float, mn: float, mx: float, tol: float) -> float:
    """0 error inside range center, grows gradually outside range by tolerance."""
    target = _range_mid(mn, mx)
    return abs(v - target) / max(tol, 1e-6)


def _fit_class_from_ease(ease_cm: float) -> str:
    if ease_cm < -2.0:
        return "Tight"
    if ease_cm < 4.0:
        return "Perfect"
    if ease_cm < 9.0:
        return "Relaxed"
    return "Loose"


def get_size_recommendation(body_measurements: Dict) -> Dict:
    """
    Get clothing size recommendation based on body measurements.

    Args:
        body_measurements: Dict containing:
            - shoulder_width: shoulder width in pixels
            - torso_height: torso height in pixels
            - chest_cm (optional): chest measurement in cm
            - shoulder_width_cm (optional): shoulder width in cm
            - torso_length_cm (optional): torso length in cm

    Returns:
        Dict containing:
            - recommended_size: Best fit size (S/M/L/XL/XXL)
            - confidence: Confidence score (0.0-1.0)
            - all_sizes: Dict with fit scores for all sizes
            - measurements_cm: Converted measurements in cm
    """
    logger.debug("Calculating size recommendation...")

    # Convert or read measurements in cm
    measurements_cm = {}

    # Shoulder width
    if "shoulder_width_cm" in body_measurements:
        measurements_cm["shoulder_width"] = body_measurements["shoulder_width_cm"]
    elif "shoulder_width" in body_measurements:
        measurements_cm["shoulder_width"] = pixels_to_cm(
            body_measurements["shoulder_width"], "shoulder_width"
        )
    else:
        measurements_cm["shoulder_width"] = 45.0  # Default fallback

    # Chest: prefer circumference if available
    if "chest_circumference_cm" in body_measurements:
        measurements_cm["chest_width"] = float(body_measurements["chest_circumference_cm"])
    elif "chest_cm" in body_measurements:
        measurements_cm["chest_width"] = float(body_measurements["chest_cm"])
    elif "chest_width" in body_measurements:
        measurements_cm["chest_width"] = pixels_to_cm(
            body_measurements["chest_width"], "chest_width"
        )
        # Waist: prefer circumference if available
        if "waist_circumference_cm" in body_measurements:
            measurements_cm["waist_width"] = float(body_measurements["waist_circumference_cm"])
        elif "waist_cm" in body_measurements:
            measurements_cm["waist_width"] = float(body_measurements["waist_cm"])
        else:
            # torso proxy fallback
            measurements_cm["waist_width"] = measurements_cm["chest_width"] * 0.86

    else:
        # Estimate chest as ~1.15x shoulder width
        measurements_cm["chest_width"] = measurements_cm["shoulder_width"] * 1.15

    # Torso length
    if "torso_length_cm" in body_measurements:
        measurements_cm["torso_length"] = body_measurements["torso_length_cm"]
    elif "torso_height" in body_measurements:
        measurements_cm["torso_length"] = pixels_to_cm(
            body_measurements["torso_height"], "torso_length"
        )
    else:
        measurements_cm["torso_length"] = 68.0  # Default fallback

    logger.debug(f"Measurements (cm): {measurements_cm}")

    # Decision layer: weighted fit scoring by normalized measurement errors
    size_scores = {}
    fit_by_size = {}

    # Per-metric reliability weighting from inference layer if present
    mconf = body_measurements.get("measurement_confidence") or {}
    conf_chest = float(mconf.get("chest", 0.8))
    conf_waist = float(mconf.get("waist", 0.7))
    conf_shoulder = float(mconf.get("shoulder", 0.85))
    conf_torso = float(mconf.get("torso", 0.75))

    base_weights = {
        "chest": 0.38,
        "waist": 0.27,
        "shoulder": 0.22,
        "torso": 0.13,
    }
    weighted_sum = (
        base_weights["chest"] * max(conf_chest, 0.35)
        + base_weights["waist"] * max(conf_waist, 0.35)
        + base_weights["shoulder"] * max(conf_shoulder, 0.35)
        + base_weights["torso"] * max(conf_torso, 0.35)
    )

    for size, ranges in SIZE_CHART.items():
        e_chest = _norm_error(measurements_cm["chest_width"], ranges["chest_min"], ranges["chest_max"], tol=8.0)
        e_waist = _norm_error(measurements_cm["waist_width"], ranges["waist_min"], ranges["waist_max"], tol=9.0)
        e_shoulder = _norm_error(measurements_cm["shoulder_width"], ranges["shoulder_min"], ranges["shoulder_max"], tol=4.0)
        e_torso = _norm_error(measurements_cm["torso_length"], ranges["torso_length_min"], ranges["torso_length_max"], tol=6.0)

        weighted_error = (
            base_weights["chest"] * max(conf_chest, 0.35) * e_chest
            + base_weights["waist"] * max(conf_waist, 0.35) * e_waist
            + base_weights["shoulder"] * max(conf_shoulder, 0.35) * e_shoulder
            + base_weights["torso"] * max(conf_torso, 0.35) * e_torso
        ) / max(weighted_sum, 1e-6)

        # Score in [0,1], where 1 is best fit
        score = _clamp01(1.0 - (weighted_error / 1.25))
        size_scores[size] = score

        chest_mid = _range_mid(ranges["chest_min"], ranges["chest_max"])
        chest_ease = chest_mid - measurements_cm["chest_width"]
        fit_by_size[size] = {
            "chest_ease_cm": round(chest_ease, 2),
            "fit_class": _fit_class_from_ease(chest_ease),
            "weighted_error": round(weighted_error, 4),
        }

    # Find best fit
    best_size = max(size_scores.keys(), key=lambda k: size_scores[k])
    confidence = float(size_scores[best_size])

    logger.info(f"Recommended size: {best_size} (confidence: {confidence:.2f})")

    return {
        "recommended_size": best_size,
        "confidence": confidence,
        "all_sizes": size_scores,
        "fit_by_size": fit_by_size,
        "fit_classification": fit_by_size[best_size]["fit_class"],
        "measurements_cm": measurements_cm,
        "size_chart": SIZE_CHART[best_size]  # Include the size chart for the recommended size
    }

def format_size_recommendation(size_rec: Dict) -> str:
    """
    Format size recommendation as a readable string.

    Args:
        size_rec: Size recommendation dict from get_size_recommendation()

    Returns:
        Formatted string for display
    """
    size = size_rec["recommended_size"]
    confidence = size_rec["confidence"]
    measurements = size_rec["measurements_cm"]

    confidence_text = ""
    if confidence > 0.8:
        confidence_text = "Excellent fit"
    elif confidence > 0.6:
        confidence_text = "Good fit"
    elif confidence > 0.4:
        confidence_text = "Reasonable fit"
    else:
        confidence_text = "Check measurements"

        fit_class = size_rec.get("fit_classification", "Perfect")
        return (f"Size: {size} ({fit_class}, {confidence_text})\n"
            f"Chest: {measurements['chest_width']:.0f}cm | "
            f"Waist: {measurements.get('waist_width', 0):.0f}cm | "
            f"Shoulders: {measurements['shoulder_width']:.0f}cm | "
            f"Length: {measurements['torso_length']:.0f}cm")

def get_size_alternatives(size_rec: Dict, threshold: float = 0.6) -> list:
    """
    Get alternative sizes that also fit well.

    Args:
        size_rec: Size recommendation dict from get_size_recommendation()
        threshold: Minimum score to consider as alternative

    Returns:
        List of alternative sizes with scores above threshold
    """
    alternatives = []
    recommended = size_rec["recommended_size"]

    for size, score in size_rec["all_sizes"].items():
        if size != recommended and score >= threshold:
            alternatives.append({"size": size, "score": score})

    # Sort by score descending
    alternatives.sort(key=lambda x: x["score"], reverse=True)

    return alternatives
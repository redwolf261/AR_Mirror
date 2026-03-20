"""
AR Mirror - Size Recommendation Engine

Provides clothing size recommendations (S/M/L/XL/XXL) based on real-time body measurements.
"""

import logging
from typing import Optional, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Standard clothing size charts (in centimeters)
# Based on average US/EU sizing standards
SIZE_CHART = {
    "XS": {
        "chest_min": 81, "chest_max": 86,
        "shoulder_min": 38, "shoulder_max": 41,
        "torso_length_min": 60, "torso_length_max": 65,
    },
    "S": {
        "chest_min": 86, "chest_max": 91,
        "shoulder_min": 41, "shoulder_max": 44,
        "torso_length_min": 65, "torso_length_max": 68,
    },
    "M": {
        "chest_min": 91, "chest_max": 97,
        "shoulder_min": 44, "shoulder_max": 47,
        "torso_length_min": 68, "torso_length_max": 71,
    },
    "L": {
        "chest_min": 97, "chest_max": 102,
        "shoulder_min": 47, "shoulder_max": 50,
        "torso_length_min": 71, "torso_length_max": 74,
    },
    "XL": {
        "chest_min": 102, "chest_max": 107,
        "shoulder_min": 50, "shoulder_max": 53,
        "torso_length_min": 74, "torso_length_max": 77,
    },
    "XXL": {
        "chest_min": 107, "chest_max": 112,
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

    # Convert pixel measurements to cm if not already provided
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

    # Chest width (estimate from shoulder width if not provided)
    if "chest_cm" in body_measurements:
        measurements_cm["chest_width"] = body_measurements["chest_cm"]
    elif "chest_width" in body_measurements:
        measurements_cm["chest_width"] = pixels_to_cm(
            body_measurements["chest_width"], "chest_width"
        )
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

    # Calculate fit scores for each size
    size_scores = {}

    for size, ranges in SIZE_CHART.items():
        scores = []

        # Chest fit score
        chest_cm = measurements_cm["chest_width"]
        if ranges["chest_min"] <= chest_cm <= ranges["chest_max"]:
            chest_score = 1.0
        else:
            # Calculate distance from range
            if chest_cm < ranges["chest_min"]:
                distance = ranges["chest_min"] - chest_cm
            else:
                distance = chest_cm - ranges["chest_max"]
            chest_score = max(0.0, 1.0 - distance / 10.0)  # Penalty per cm outside range
        scores.append(chest_score * 0.5)  # 50% weight

        # Shoulder fit score
        shoulder_cm = measurements_cm["shoulder_width"]
        if ranges["shoulder_min"] <= shoulder_cm <= ranges["shoulder_max"]:
            shoulder_score = 1.0
        else:
            if shoulder_cm < ranges["shoulder_min"]:
                distance = ranges["shoulder_min"] - shoulder_cm
            else:
                distance = shoulder_cm - ranges["shoulder_max"]
            shoulder_score = max(0.0, 1.0 - distance / 5.0)
        scores.append(shoulder_score * 0.3)  # 30% weight

        # Torso length fit score
        torso_cm = measurements_cm["torso_length"]
        if ranges["torso_length_min"] <= torso_cm <= ranges["torso_length_max"]:
            torso_score = 1.0
        else:
            if torso_cm < ranges["torso_length_min"]:
                distance = ranges["torso_length_min"] - torso_cm
            else:
                distance = torso_cm - ranges["torso_length_max"]
            torso_score = max(0.0, 1.0 - distance / 8.0)
        scores.append(torso_score * 0.2)  # 20% weight

        # Overall size score
        size_scores[size] = sum(scores)
        logger.debug(f"Size {size}: score {size_scores[size]:.2f}")

    # Find best fit
    best_size = max(size_scores.keys(), key=lambda k: size_scores[k])
    confidence = size_scores[best_size]

    logger.info(f"Recommended size: {best_size} (confidence: {confidence:.2f})")

    return {
        "recommended_size": best_size,
        "confidence": confidence,
        "all_sizes": size_scores,
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

    return (f"Size: {size} ({confidence_text})\n"
            f"Chest: {measurements['chest_width']:.0f}cm | "
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
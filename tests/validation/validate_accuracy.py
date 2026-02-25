"""
Validation script to test measurement accuracy against manual measurements
Used for calibration and bias correction analysis
"""

import json
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationRecord:
    person_id: str
    manual_shoulder_cm: float
    manual_chest_cm: float
    manual_torso_cm: float
    system_shoulder_cm: float
    system_chest_cm: float
    system_torso_cm: float
    confidence: float


def compute_error_metrics(records: List[ValidationRecord]) -> Dict:
    shoulder_errors = [r.system_shoulder_cm - r.manual_shoulder_cm for r in records]
    chest_errors = [r.system_chest_cm - r.manual_chest_cm for r in records]
    torso_errors = [r.system_torso_cm - r.manual_torso_cm for r in records]
    
    return {
        'shoulder': {
            'mean_error': np.mean(shoulder_errors),
            'std_error': np.std(shoulder_errors),
            'mae': np.mean(np.abs(shoulder_errors)),
            'max_error': np.max(np.abs(shoulder_errors))
        },
        'chest': {
            'mean_error': np.mean(chest_errors),
            'std_error': np.std(chest_errors),
            'mae': np.mean(np.abs(chest_errors)),
            'max_error': np.max(np.abs(chest_errors))
        },
        'torso': {
            'mean_error': np.mean(torso_errors),
            'std_error': np.std(torso_errors),
            'mae': np.mean(np.abs(torso_errors)),
            'max_error': np.max(np.abs(torso_errors))
        }
    }


def load_validation_data(filepath: str) -> List[ValidationRecord]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return [ValidationRecord(**record) for record in data]


def generate_bias_correction_table(records: List[ValidationRecord]) -> Dict:
    corrections = {
        'global': {
            'shoulder_bias_cm': -np.mean([r.system_shoulder_cm - r.manual_shoulder_cm for r in records]),
            'chest_bias_cm': -np.mean([r.system_chest_cm - r.manual_chest_cm for r in records]),
            'torso_bias_cm': -np.mean([r.system_torso_cm - r.manual_torso_cm for r in records])
        }
    }
    
    return corrections


def main():
    validation_file = "validation_data.json"
    
    if not Path(validation_file).exists():
        print(f"Creating sample validation file: {validation_file}")
        sample_data = [
            {
                "person_id": "P001",
                "manual_shoulder_cm": 42.0,
                "manual_chest_cm": 46.0,
                "manual_torso_cm": 62.0,
                "system_shoulder_cm": 43.2,
                "system_chest_cm": 47.1,
                "system_torso_cm": 61.5,
                "confidence": 0.88
            }
        ]
        with open(validation_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Sample file created. Add real validation data and re-run.")
        return
    
    records = load_validation_data(validation_file)
    
    if len(records) < 10:
        print(f"Warning: Only {len(records)} validation records found.")
        print("Recommended: 50+ records for reliable bias correction")
        print()
    
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total records: {len(records)}")
    print()
    
    metrics = compute_error_metrics(records)
    
    for part, stats in metrics.items():
        print(f"{part.upper()} measurements:")
        print(f"  Mean error:     {stats['mean_error']:+.2f} cm")
        print(f"  Std deviation:  {stats['std_error']:.2f} cm")
        print(f"  Mean abs error: {stats['mae']:.2f} cm")
        print(f"  Max abs error:  {stats['max_error']:.2f} cm")
        print()
    
    corrections = generate_bias_correction_table(records)
    
    print("BIAS CORRECTION TABLE:")
    print(json.dumps(corrections, indent=2))
    print()
    
    with open('bias_corrections.json', 'w') as f:
        json.dump(corrections, f, indent=2)
    
    print("Bias corrections saved to: bias_corrections.json")
    print()
    
    acceptable_mae = 2.5
    all_acceptable = all(
        metrics[part]['mae'] < acceptable_mae 
        for part in ['shoulder', 'chest', 'torso']
    )
    
    if all_acceptable:
        print("✅ System accuracy within acceptable range (<2.5cm MAE)")
    else:
        print("⚠️  System accuracy needs improvement (>2.5cm MAE)")
        print("   Collect more data or adjust measurement logic")


if __name__ == "__main__":
    main()

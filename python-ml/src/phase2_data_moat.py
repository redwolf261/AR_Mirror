"""
PHASE 2: Data Moat - Per-SKU Fit Learning & Demographic Bias Correction

This module learns from actual purchase outcomes and demographic patterns
to improve fit predictions beyond raw measurements.

Requires: 100+ measurements with purchase outcome data (Days 31-90)
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class SKUCorrection:
    """Learned correction factors for a specific SKU"""
    sku: str
    num_samples: int
    shoulder_bias: float  # Mean error: system - actual
    chest_bias: float
    torso_bias: float
    shoulder_tolerance: float  # Adjusted ease factor
    chest_tolerance: float
    torso_tolerance: float
    confidence_multiplier: float

@dataclass
class DemographicCluster:
    """Body measurement cluster for demographic correction"""
    cluster_id: int
    num_samples: int
    shoulder_range: tuple
    chest_range: tuple
    torso_range: tuple
    head_scale_offset: float  # Correction to 23cm assumption
    confidence_adjust: float

class DataMoat:
    """PHASE 2: Learn from production data to build competitive advantage"""
    
    def __init__(self, data_dir="validation_data"):
        self.data_dir = Path(data_dir)
        self.sku_corrections = {}
        self.demographic_clusters = []
        self.correction_file = self.data_dir / "learned_corrections.json"
        
        # Load existing corrections if available
        if self.correction_file.exists():
            self.load_corrections()
    
    def learn_from_validation_data(self, validation_files: List[str]):
        """PHASE 2A: Learn SKU-specific corrections from validation data"""
        print("=" * 70)
        print("PHASE 2A: Learning Per-SKU Corrections")
        print("=" * 70)
        
        # Aggregate data by SKU
        sku_data = {}
        
        for file_path in validation_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for subject in data.get('subjects', []):
                # Extract SKU from system measurement (if available)
                sku = "SKU-001"  # Default for MVP
                
                if sku not in sku_data:
                    sku_data[sku] = {
                        'shoulder_errors': [],
                        'chest_errors': [],
                        'torso_errors': []
                    }
                
                errors = subject.get('errors', {})
                sku_data[sku]['shoulder_errors'].append(errors.get('shoulder', 0))
                sku_data[sku]['chest_errors'].append(errors.get('chest', 0))
                sku_data[sku]['torso_errors'].append(errors.get('torso', 0))
        
        # Calculate corrections per SKU
        for sku, data in sku_data.items():
            if len(data['shoulder_errors']) < 10:
                print(f"\nSKU {sku}: Insufficient data ({len(data['shoulder_errors'])} samples)")
                print("  Need 10+ samples to learn corrections")
                continue
            
            # Calculate systematic bias
            shoulder_bias = np.mean(data['shoulder_errors'])
            chest_bias = np.mean(data['chest_errors'])
            torso_bias = np.mean(data['torso_errors'])
            
            # Calculate variance to adjust tolerance
            shoulder_std = np.std(data['shoulder_errors'])
            chest_std = np.std(data['chest_errors'])
            torso_std = np.std(data['torso_errors'])
            
            # Adjust ease factors based on learned variance
            # If system consistently under-predicts (negative bias), increase tolerance
            shoulder_tolerance = 2.0 - shoulder_bias if shoulder_bias < 0 else 2.0
            chest_tolerance = 4.0 - chest_bias if chest_bias < 0 else 4.0
            torso_tolerance = 3.0 - torso_bias if torso_bias < 0 else 3.0
            
            # Adjust confidence based on consistency
            # Lower std = more confident predictions
            confidence_multiplier = 1.0 - (shoulder_std + chest_std + torso_std) / 30.0
            confidence_multiplier = max(0.8, min(1.2, confidence_multiplier))
            
            correction = SKUCorrection(
                sku=sku,
                num_samples=len(data['shoulder_errors']),
                shoulder_bias=shoulder_bias,
                chest_bias=chest_bias,
                torso_bias=torso_bias,
                shoulder_tolerance=shoulder_tolerance,
                chest_tolerance=chest_tolerance,
                torso_tolerance=torso_tolerance,
                confidence_multiplier=float(confidence_multiplier)
            )
            
            self.sku_corrections[sku] = correction
            
            print(f"\n{sku} ({correction.num_samples} samples):")
            print(f"  Bias: Shoulder={shoulder_bias:+.2f}cm, "
                  f"Chest={chest_bias:+.2f}cm, Torso={torso_bias:+.2f}cm")
            print(f"  Adjusted tolerances: Shoulder={shoulder_tolerance:.1f}cm, "
                  f"Chest={chest_tolerance:.1f}cm, Torso={torso_tolerance:.1f}cm")
            print(f"  Confidence multiplier: {confidence_multiplier:.2f}")
        
        self.save_corrections()
        print("\n✓ SKU corrections learned and saved")
    
    def learn_demographic_clusters(self, validation_files: List[str], num_clusters=3):
        """PHASE 2B: Identify body type clusters for demographic correction"""
        print("\n" + "=" * 70)
        print("PHASE 2B: Learning Demographic Clusters")
        print("=" * 70)
        
        # Collect measurement ratios
        subjects_data = []
        
        for file_path in validation_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for subject in data.get('subjects', []):
                manual = subject.get('manual_measurement', {})
                system = subject.get('system_measurement', {})
                errors = subject.get('errors', {})
                
                if manual and system:
                    subjects_data.append({
                        'shoulder': manual['shoulder_cm'],
                        'chest': manual['chest_cm'],
                        'torso': manual['torso_cm'],
                        'shoulder_error': errors.get('shoulder', 0),
                        'chest_error': errors.get('chest', 0),
                        'torso_error': errors.get('torso', 0)
                    })
        
        if len(subjects_data) < 30:
            print(f"\nInsufficient data for clustering ({len(subjects_data)} subjects)")
            print("  Need 30+ diverse subjects for demographic analysis")
            return
        
        # Simple k-means clustering by body proportions
        measurements = np.array([[s['shoulder'], s['chest'], s['torso']] 
                                for s in subjects_data])
        
        # Normalize measurements
        mean = measurements.mean(axis=0)
        std = measurements.std(axis=0)
        normalized = (measurements - mean) / std
        
        # Simple k-means (could use sklearn in production)
        centroids = self._simple_kmeans(normalized, k=num_clusters)
        
        # Assign clusters and calculate per-cluster corrections
        for cluster_id in range(num_clusters):
            cluster_subjects = []
            cluster_errors = {'shoulder': [], 'chest': [], 'torso': []}
            
            for i, point in enumerate(normalized):
                # Find nearest centroid
                distances = [np.linalg.norm(point - c) for c in centroids]
                if np.argmin(distances) == cluster_id:
                    cluster_subjects.append(subjects_data[i])
                    cluster_errors['shoulder'].append(subjects_data[i]['shoulder_error'])
                    cluster_errors['chest'].append(subjects_data[i]['chest_error'])
                    cluster_errors['torso'].append(subjects_data[i]['torso_error'])
            
            if len(cluster_subjects) < 5:
                continue
            
            # Calculate cluster statistics
            shoulders = [s['shoulder'] for s in cluster_subjects]
            chests = [s['chest'] for s in cluster_subjects]
            torsos = [s['torso'] for s in cluster_subjects]
            
            # Head scale offset: if cluster consistently has error, adjust scale
            mean_error = np.mean([abs(e) for e in cluster_errors['shoulder']])
            head_scale_offset = -np.mean(cluster_errors['shoulder']) * 0.5  # Conservative
            
            cluster = DemographicCluster(
                cluster_id=cluster_id,
                num_samples=len(cluster_subjects),
                shoulder_range=(min(shoulders), max(shoulders)),
                chest_range=(min(chests), max(chests)),
                torso_range=(min(torsos), max(torsos)),
                head_scale_offset=float(head_scale_offset),
                confidence_adjust=1.0 if mean_error < 2.0 else 0.9
            )
            
            self.demographic_clusters.append(cluster)
            
            print(f"\nCluster {cluster_id} ({cluster.num_samples} subjects):")
            print(f"  Shoulder: {cluster.shoulder_range[0]:.1f}-{cluster.shoulder_range[1]:.1f}cm")
            print(f"  Chest: {cluster.chest_range[0]:.1f}-{cluster.chest_range[1]:.1f}cm")
            print(f"  Torso: {cluster.torso_range[0]:.1f}-{cluster.torso_range[1]:.1f}cm")
            print(f"  Head scale offset: {cluster.head_scale_offset:+.2f}cm")
            print(f"  Confidence adjust: {cluster.confidence_adjust:.2f}")
        
        self.save_corrections()
        print("\n✓ Demographic clusters identified and saved")
    
    def _simple_kmeans(self, data: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """Simple k-means implementation"""
        # Initialize centroids randomly
        indices = np.random.choice(len(data), k, replace=False)
        centroids = data[indices]
        
        for _ in range(max_iter):
            # Assign points to nearest centroid
            distances = np.array([[np.linalg.norm(point - c) for c in centroids] 
                                 for point in data])
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([data[assignments == i].mean(axis=0) 
                                     for i in range(k)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return centroids
    
    def apply_corrections(self, sku: str, measurements: Dict, 
                         system_measurements: Dict) -> Dict:
        """Apply learned corrections to measurements"""
        corrected = measurements.copy()
        
        # Apply SKU-specific corrections
        if sku in self.sku_corrections:
            correction = self.sku_corrections[sku]
            corrected['shoulder_cm'] -= correction.shoulder_bias
            corrected['chest_cm'] -= correction.chest_bias
            corrected['torso_cm'] -= correction.torso_bias
            corrected['confidence'] *= correction.confidence_multiplier
        
        # Apply demographic cluster corrections
        cluster = self._find_cluster(measurements)
        if cluster:
            # Adjust scale factor retrospectively
            # This is simplified - in production would re-run measurement estimation
            scale_adjust = 1.0 + (cluster.head_scale_offset / 23.0)
            corrected['shoulder_cm'] *= scale_adjust
            corrected['chest_cm'] *= scale_adjust
            corrected['torso_cm'] *= scale_adjust
            corrected['confidence'] *= cluster.confidence_adjust
        
        return corrected
    
    def _find_cluster(self, measurements: Dict) -> Optional[DemographicCluster]:
        """Find which demographic cluster measurements belong to"""
        shoulder = measurements.get('shoulder_cm', 0)
        chest = measurements.get('chest_cm', 0)
        torso = measurements.get('torso_cm', 0)
        
        for cluster in self.demographic_clusters:
            if (cluster.shoulder_range[0] <= shoulder <= cluster.shoulder_range[1] and
                cluster.chest_range[0] <= chest <= cluster.chest_range[1] and
                cluster.torso_range[0] <= torso <= cluster.torso_range[1]):
                return cluster
        
        return None
    
    def save_corrections(self):
        """Save learned corrections to file"""
        data = {
            'sku_corrections': {k: asdict(v) for k, v in self.sku_corrections.items()},
            'demographic_clusters': [asdict(c) for c in self.demographic_clusters]
        }
        
        with open(self.correction_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_corrections(self):
        """Load previously learned corrections"""
        with open(self.correction_file, 'r') as f:
            data = json.load(f)
        
        self.sku_corrections = {
            k: SKUCorrection(**v) for k, v in data.get('sku_corrections', {}).items()
        }
        
        self.demographic_clusters = [
            DemographicCluster(**c) for c in data.get('demographic_clusters', [])
        ]

def main():
    """Demo: Learn from validation data"""
    print("=" * 70)
    print("PHASE 2: DATA MOAT - Learning from Validation Data")
    print("=" * 70)
    
    moat = DataMoat()
    
    # Find validation files
    validation_dir = Path("validation_data")
    validation_files = list(validation_dir.glob("validation_session_*.json"))
    
    if not validation_files:
        print("\nNo validation data found!")
        print("Run collect_validation_data.py first to gather ground truth")
        return 1
    
    print(f"\nFound {len(validation_files)} validation session(s)")
    
    # Learn corrections
    moat.learn_from_validation_data([str(f) for f in validation_files])
    moat.learn_demographic_clusters([str(f) for f in validation_files])
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print("\nLearned corrections saved to: learned_corrections.json")
    print("\nNext: Integrate corrections into sizing_pipeline.py")
    print("      Update MeasurementEstimator.estimate() to call moat.apply_corrections()")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

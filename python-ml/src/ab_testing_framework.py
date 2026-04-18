"""
A/B Testing Framework
Test different algorithms, thresholds, and corrections simultaneously
PHASE 2C implementation
"""
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict

@dataclass
class ABTestConfig:
    test_name: str
    variants: List[str]
    start_date: str
    end_date: Optional[str]
    description: str
    active: bool

@dataclass
class ABTestResult:
    session_id: str
    test_name: str
    variant: str
    predicted_fit: str
    actual_fit: Optional[str]
    accuracy: Optional[float]
    confidence: float
    timestamp: str
    metadata: Dict

class ABTestingFramework:
    """A/B testing for measurement algorithms and fit decisions"""
    
    def __init__(self, output_dir: str = "ab_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.results: List[ABTestResult] = []
    
    def create_test(self, test_name: str, variants: List[str], 
                   description: str) -> ABTestConfig:
        """Create a new A/B test"""
        
        config = ABTestConfig(
            test_name=test_name,
            variants=variants,
            start_date=datetime.now().isoformat(),
            end_date=None,
            description=description,
            active=True
        )
        
        self.active_tests[test_name] = config
        print(f"✓ Created A/B test: {test_name}")
        print(f"  Variants: {', '.join(variants)}")
        print(f"  Description: {description}")
        
        return config
    
    def assign_variant(self, user_id: str, test_name: str) -> str:
        """
        Assign user to a variant consistently
        Uses hash of user_id + test_name for deterministic assignment
        """
        
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        config = self.active_tests[test_name]
        
        # Hash user_id + test_name for consistent assignment
        hash_input = f"{user_id}_{test_name}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Assign to variant based on hash modulo
        variant_index = hash_value % len(config.variants)
        variant = config.variants[variant_index]
        
        return variant
    
    def record_result(self, session_id: str, test_name: str, variant: str,
                     predicted_fit: str, actual_fit: Optional[str] = None,
                     confidence: float = 0.0, metadata: Optional[Dict] = None):
        """Record an A/B test result"""
        
        accuracy = None
        if actual_fit is not None:
            accuracy = 1.0 if predicted_fit == actual_fit else 0.0
        
        result = ABTestResult(
            session_id=session_id,
            test_name=test_name,
            variant=variant,
            predicted_fit=predicted_fit,
            actual_fit=actual_fit,
            accuracy=accuracy,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.results.append(result)
    
    def analyze_test(self, test_name: str, min_samples: int = 30) -> Dict:
        """
        Analyze A/B test results with statistical significance
        
        Returns:
        {
            'test_name': str,
            'variants': {
                'control': {'sample_size': int, 'accuracy': float, ...},
                'treatment_a': {...},
                ...
            },
            'winner': str,
            'confidence': float,
            'recommendation': str
        }
        """
        
        if test_name not in self.active_tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        # Filter results for this test
        test_results = [r for r in self.results if r.test_name == test_name]
        
        if len(test_results) < min_samples:
            return {
                'test_name': test_name,
                'status': 'insufficient_data',
                'samples': len(test_results),
                'required': min_samples
            }
        
        # Group by variant
        variant_results = defaultdict(list)
        for result in test_results:
            variant_results[result.variant].append(result)
        
        # Calculate metrics per variant
        analysis = {
            'test_name': test_name,
            'total_samples': len(test_results),
            'variants': {}
        }
        
        for variant, results in variant_results.items():
            # Calculate accuracy (only for results with ground truth)
            results_with_truth = [r for r in results if r.actual_fit is not None]
            
            if results_with_truth:
                accuracy = np.mean([r.accuracy for r in results_with_truth if r.accuracy is not None])
                avg_confidence = np.mean([r.confidence for r in results_with_truth])
                
                # Calculate standard error
                accuracies = [r.accuracy for r in results_with_truth if r.accuracy is not None]
                std_error = np.std(accuracies) / np.sqrt(len(accuracies))
                
                analysis['variants'][variant] = {
                    'sample_size': len(results),
                    'validated_samples': len(results_with_truth),
                    'accuracy': float(accuracy),
                    'std_error': float(std_error),
                    'avg_confidence': float(avg_confidence),
                    'confidence_interval_95': (
                        float(accuracy - 1.96 * std_error),
                        float(accuracy + 1.96 * std_error)
                    )
                }
            else:
                analysis['variants'][variant] = {
                    'sample_size': len(results),
                    'validated_samples': 0,
                    'accuracy': None,
                    'status': 'awaiting_ground_truth'
                }
        
        # Determine winner (if control exists)
        if 'control' in analysis['variants'] and len(analysis['variants']) > 1:
            control_accuracy = analysis['variants']['control'].get('accuracy', 0)
            
            best_variant = 'control'
            best_accuracy = control_accuracy
            best_improvement = 0.0
            
            for variant, metrics in analysis['variants'].items():
                if variant == 'control' or metrics.get('accuracy') is None:
                    continue
                
                variant_accuracy = metrics['accuracy']
                improvement = (variant_accuracy - control_accuracy) / control_accuracy * 100
                
                if variant_accuracy > best_accuracy:
                    best_variant = variant
                    best_accuracy = variant_accuracy
                    best_improvement = improvement
            
            # Calculate statistical significance (two-proportion z-test)
            significance = self._calculate_significance(
                analysis['variants']['control'],
                analysis['variants'][best_variant]
            )
            
            analysis['winner'] = best_variant
            analysis['improvement_pct'] = float(best_improvement)
            analysis['statistical_significance'] = significance
            
            # Recommendation
            if significance > 0.95 and best_improvement > 5.0:
                analysis['recommendation'] = f"DEPLOY {best_variant}: Significant improvement ({best_improvement:.1f}%)"
            elif significance > 0.90 and best_improvement > 3.0:
                analysis['recommendation'] = f"CONDITIONAL: {best_variant} shows promise, collect more data"
            else:
                analysis['recommendation'] = "NO CHANGE: No significant improvement detected"
        
        return analysis
    
    def _calculate_significance(self, control: Dict, treatment: Dict) -> float:
        """Calculate statistical significance using two-proportion z-test"""
        
        if control.get('accuracy') is None or treatment.get('accuracy') is None:
            return 0.0
        
        n1 = control['validated_samples']
        n2 = treatment['validated_samples']
        
        if n1 < 10 or n2 < 10:
            return 0.0  # Insufficient data
        
        p1 = control['accuracy']
        p2 = treatment['accuracy']
        
        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        if se == 0:
            return 0.0
        
        # Z-score
        z = (p2 - p1) / se
        
        # Convert to p-value (one-tailed)
        from scipy import stats
        p_value = 1 - stats.norm.cdf(z)
        
        # Return confidence level (1 - p_value)
        return float(1 - p_value)
    
    def end_test(self, test_name: str):
        """Mark a test as complete"""
        if test_name in self.active_tests:
            self.active_tests[test_name].active = False
            self.active_tests[test_name].end_date = datetime.now().isoformat()
            print(f"✓ Ended test: {test_name}")
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save A/B test results"""
        if filename is None:
            filename = f"ab_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'active_tests': {name: asdict(config) for name, config in self.active_tests.items()},
            'results': [asdict(r) for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {len(self.results)} results to {filepath}")
        return filepath
    
    def load_results(self, filename: str):
        """Load A/B test results from file"""
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            print(f"⚠ Results file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.active_tests = {
            name: ABTestConfig(**config_data)
            for name, config_data in data['active_tests'].items()
        }
        
        self.results = [ABTestResult(**r_data) for r_data in data['results']]
        
        print(f"✓ Loaded {len(self.results)} results from {filepath}")


def main():
    """Demo: A/B testing framework"""
    print("=" * 70)
    print("A/B TESTING FRAMEWORK DEMO")
    print("=" * 70)
    
    framework = ABTestingFramework()
    
    # Test 1: SKU corrections vs baseline
    print("\n1. Creating Test: SKU Corrections")
    test1 = framework.create_test(
        test_name="sku_corrections_v1",
        variants=["control", "sku_corrected"],
        description="Test impact of per-SKU fit tolerance adjustments"
    )
    
    # Test 2: Lighting enhancement
    print("\n2. Creating Test: Lighting Enhancement")
    test2 = framework.create_test(
        test_name="lighting_clahe_v1",
        variants=["control", "clahe_enhanced"],
        description="Test CLAHE enhancement for low-light conditions"
    )
    
    # Simulate user assignments
    print("\n3. Simulating user assignments...")
    users = [f"user_{i}" for i in range(100)]
    
    for user_id in users[:5]:
        variant1 = framework.assign_variant(user_id, "sku_corrections_v1")
        variant2 = framework.assign_variant(user_id, "lighting_clahe_v1")
        print(f"   {user_id}: SKU={variant1}, Lighting={variant2}")
    
    # Simulate test results
    print("\n4. Simulating test results...")
    
    np.random.seed(42)
    
    for i, user_id in enumerate(users):
        # Test 1: SKU corrections
        variant = framework.assign_variant(user_id, "sku_corrections_v1")
        
        # Control: 75% accuracy
        # SKU corrected: 85% accuracy
        if variant == "control":
            predicted = "GOOD"
            actual = "GOOD" if np.random.rand() < 0.75 else np.random.choice(["TIGHT", "LOOSE"])
        else:
            predicted = "GOOD"
            actual = "GOOD" if np.random.rand() < 0.85 else np.random.choice(["TIGHT", "LOOSE"])
        
        framework.record_result(
            session_id=f"sess_{i}",
            test_name="sku_corrections_v1",
            variant=variant,
            predicted_fit=predicted,
            actual_fit=actual,
            confidence=0.8 + np.random.rand() * 0.15,
            metadata={'user_id': user_id}
        )
        
        # Test 2: Lighting enhancement
        variant = framework.assign_variant(user_id, "lighting_clahe_v1")
        
        # Control: 70% accuracy in low light
        # CLAHE: 80% accuracy in low light
        if variant == "control":
            predicted = "GOOD"
            actual = "GOOD" if np.random.rand() < 0.70 else np.random.choice(["TIGHT", "LOOSE"])
        else:
            predicted = "GOOD"
            actual = "GOOD" if np.random.rand() < 0.80 else np.random.choice(["TIGHT", "LOOSE"])
        
        framework.record_result(
            session_id=f"sess_{i}",
            test_name="lighting_clahe_v1",
            variant=variant,
            predicted_fit=predicted,
            actual_fit=actual,
            confidence=0.75 + np.random.rand() * 0.15,
            metadata={'user_id': user_id, 'lighting': 'low'}
        )
    
    print(f"   Recorded {len(framework.results)} test results")
    
    # Analyze Test 1
    print("\n5. Analyzing Test 1: SKU Corrections")
    print("=" * 70)
    analysis1 = framework.analyze_test("sku_corrections_v1")
    
    for variant, metrics in analysis1['variants'].items():
        print(f"\n   {variant.upper()}:")
        print(f"      Sample size: {metrics['sample_size']}")
        print(f"      Validated samples: {metrics['validated_samples']}")
        print(f"      Accuracy: {metrics['accuracy']:.2%}")
        print(f"      Avg confidence: {metrics['avg_confidence']:.2f}")
        print(f"      95% CI: [{metrics['confidence_interval_95'][0]:.2%}, {metrics['confidence_interval_95'][1]:.2%}]")
    
    print(f"\n   WINNER: {analysis1['winner']}")
    print(f"   Improvement: {analysis1['improvement_pct']:.1f}%")
    print(f"   Statistical significance: {analysis1['statistical_significance']:.2%}")
    print(f"\n   RECOMMENDATION: {analysis1['recommendation']}")
    
    # Analyze Test 2
    print("\n6. Analyzing Test 2: Lighting Enhancement")
    print("=" * 70)
    analysis2 = framework.analyze_test("lighting_clahe_v1")
    
    for variant, metrics in analysis2['variants'].items():
        print(f"\n   {variant.upper()}:")
        print(f"      Sample size: {metrics['sample_size']}")
        print(f"      Accuracy: {metrics['accuracy']:.2%}")
        print(f"      Avg confidence: {metrics['avg_confidence']:.2f}")
    
    print(f"\n   WINNER: {analysis2['winner']}")
    print(f"   Improvement: {analysis2['improvement_pct']:.1f}%")
    print(f"\n   RECOMMENDATION: {analysis2['recommendation']}")
    
    # Save results
    print("\n7. Saving test results...")
    framework.save_results()
    
    print("\n" + "=" * 70)
    print("A/B testing framework operational!")
    print("Enables: Data-driven decisions on algorithm improvements")
    print("=" * 70)


if __name__ == "__main__":
    try:
        from scipy import stats
    except ImportError:
        print("⚠ scipy not installed. Statistical significance calculation limited.")
        print("  Install with: pip install scipy")
    
    main()

"""
PHASE 3: External Testing Analysis
Computes MAE, bias, and success rate from collected validation data
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt  # type: ignore

class ValidationAnalyzer:
    """Analyzes validation data to compute key metrics"""
    
    def __init__(self, data_dir='validation_data'):
        self.data_dir = Path(data_dir)
        self.sessions = []
        self.load_data()
    
    def load_data(self):
        """Load all validation sessions"""
        
        json_files = list(self.data_dir.glob('*.json'))
        
        if not json_files:
            print(f"⚠️  No validation data found in {self.data_dir}")
            return
        
        print(f"\n📂 Loading {len(json_files)} validation sessions...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    session = json.load(f)
                    self.sessions.append(session)
            except Exception as e:
                print(f"  ⚠️  Failed to load {json_file}: {e}")
        
        print(f"  ✓ Loaded {len(self.sessions)} sessions")
    
    def compute_metrics(self):
        """Compute all validation metrics"""
        
        if not self.sessions:
            print("❌ No data to analyze")
            return None
        
        print("\n" + "="*70)
        print("PHASE 3: EXTERNAL TESTING ANALYSIS")
        print("="*70)
        
        # Extract measurements
        dimensions = ['shoulder_width_cm', 'chest_width_cm', 'torso_length_cm']
        errors = defaultdict(list)
        all_errors = []
        
        for session in self.sessions:
            manual = session.get('manual_measurements', {})
            system = session.get('system_measurements', {})
            
            for dim in dimensions:
                if dim in manual and dim in system:
                    error = abs(system[dim] - manual[dim])
                    errors[dim].append(error)
                    all_errors.append(error)
        
        if not all_errors:
            print("❌ No valid measurements found")
            return None
        
        # Overall MAE
        mae = np.mean(all_errors)
        print(f"\n📊 OVERALL METRICS")
        print(f"{'='*70}")
        print(f"{'Mean Absolute Error (MAE):':<40} {mae:.2f} cm")
        print(f"{'Standard Deviation:':<40} {np.std(all_errors):.2f} cm")
        print(f"{'Max Error:':<40} {max(all_errors):.2f} cm")
        print(f"{'Min Error:':<40} {min(all_errors):.2f} cm")
        print(f"{'Total Samples:':<40} {len(self.sessions)}")
        
        # Gate check
        print(f"\n{'='*70}")
        if mae <= 2.5:
            print("✅ PASS: MAE ≤ 2.5cm - CONTINUE TO PILOT")
        elif mae <= 3.0:
            print("⚠️  MARGINAL: 2.5 < MAE ≤ 3.0cm - INVESTIGATE & CORRECT")
        else:
            print("❌ FAIL: MAE > 3.0cm - STOP AND FIX")
        print(f"{'='*70}")
        
        # Per-dimension breakdown
        print(f"\n📏 PER-DIMENSION ANALYSIS")
        print(f"{'='*70}")
        print(f"{'Dimension':<25} {'MAE (cm)':<12} {'Samples':<10} {'Status'}")
        print(f"{'-'*70}")
        
        for dim in dimensions:
            if errors[dim]:
                dim_mae = np.mean(errors[dim])
                dim_status = "✅" if dim_mae <= 2.5 else ("⚠️ " if dim_mae <= 3.0 else "❌")
                print(f"{dim:<25} {dim_mae:<12.2f} {len(errors[dim]):<10} {dim_status}")
        
        # Success rate
        success_count = sum(1 for e in all_errors if e <= 2.5)
        success_rate = success_count / len(all_errors) * 100
        
        print(f"\n✅ SUCCESS RATE")
        print(f"{'='*70}")
        print(f"{'Measurements within 2.5cm:':<40} {success_count}/{len(all_errors)} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("✅ PASS: Success rate ≥ 80%")
        else:
            print(f"❌ FAIL: Success rate < 80% (need {0.8*len(all_errors):.0f}/{len(all_errors)})")
        
        # Bias analysis
        print(f"\n📈 BIAS ANALYSIS")
        print(f"{'='*70}")
        
        for session in self.sessions:
            manual = session.get('manual_measurements', {})
            system = session.get('system_measurements', {})
            
            for dim in dimensions:
                if dim in manual and dim in system:
                    bias = system[dim] - manual[dim]  # Positive = overestimate
                    errors[f'{dim}_bias'].append(bias)
        
        print(f"{'Dimension':<25} {'Mean Bias (cm)':<15} {'Issue'}")
        print(f"{'-'*70}")
        
        bias_issues = []
        for dim in dimensions:
            bias_key = f'{dim}_bias'
            if bias_key in errors and errors[bias_key]:
                mean_bias = np.mean(errors[bias_key])
                bias_str = f"{mean_bias:+.2f}"
                
                if abs(mean_bias) > 3.0:
                    issue = "❌ >3cm bias!"
                    bias_issues.append(dim)
                elif abs(mean_bias) > 1.5:
                    issue = "⚠️  Notable bias"
                else:
                    issue = "✅ OK"
                
                print(f"{dim:<25} {bias_str:<15} {issue}")
        
        if bias_issues:
            print(f"\n❌ Systematic bias detected in: {', '.join(bias_issues)}")
        else:
            print(f"\n✅ No systematic bias >3cm detected")
        
        # Metadata analysis
        print(f"\n🔍 FAILURE ANALYSIS")
        print(f"{'='*70}")
        
        high_error_sessions = [s for s in self.sessions 
                              if any(abs(s['system_measurements'].get(dim, 0) - 
                                        s['manual_measurements'].get(dim, 0)) > 3.0 
                                    for dim in dimensions)]
        
        print(f"{'Sessions with >3cm error:':<40} {len(high_error_sessions)}/{len(self.sessions)}")
        
        if high_error_sessions:
            print("\nHigh-error sessions:")
            for session in high_error_sessions[:5]:  # Show first 5
                subject_id = session.get('subject_id', 'unknown')
                print(f"  - {subject_id}")
                
                metadata = session.get('system_metadata', {})
                print(f"      Lighting: {metadata.get('lighting_quality', 'N/A')}")
                print(f"      Pose: {metadata.get('pose_quality', 'N/A')}")
                print(f"      Confidence: {metadata.get('confidence', 'N/A')}")
        
        # Final verdict
        print(f"\n{'='*70}")
        print("VALIDATION VERDICT")
        print(f"{'='*70}")
        
        gates_passed = []
        gates_failed = []
        
        if mae <= 2.5:
            gates_passed.append("MAE ≤ 2.5cm")
        else:
            gates_failed.append(f"MAE = {mae:.2f}cm > 2.5cm")
        
        if success_rate >= 80:
            gates_passed.append("Success rate ≥ 80%")
        else:
            gates_failed.append(f"Success rate = {success_rate:.1f}% < 80%")
        
        if not bias_issues:
            gates_passed.append("No bias >3cm")
        else:
            gates_failed.append(f"Bias >3cm in {len(bias_issues)} dimensions")
        
        print(f"\nPassed gates ({len(gates_passed)}/3):")
        for gate in gates_passed:
            print(f"  ✅ {gate}")
        
        if gates_failed:
            print(f"\nFailed gates ({len(gates_failed)}/3):")
            for gate in gates_failed:
                print(f"  ❌ {gate}")
        
        print(f"\n{'='*70}")
        if len(gates_passed) == 3:
            print("✅ GO: System ready for retail pilot")
        elif mae <= 3.0 and success_rate >= 70:
            print("⚠️  CONDITIONAL: Investigate issues before pilot")
        else:
            print("❌ NO-GO: Fix critical issues before pilot")
        print(f"{'='*70}\n")
        
        return {
            'mae': mae,
            'success_rate': success_rate,
            'per_dimension': {dim: np.mean(errors[dim]) for dim in dimensions if errors[dim]},
            'bias': {dim: np.mean(errors[f'{dim}_bias']) for dim in dimensions if f'{dim}_bias' in errors},
            'gates_passed': gates_passed,
            'gates_failed': gates_failed,
            'verdict': 'GO' if len(gates_passed) == 3 else ('CONDITIONAL' if mae <= 3.0 else 'NO-GO')
        }
    
    def plot_results(self, save_path='validation_analysis.png'):
        """Generate visualization of results"""
        
        if not self.sessions:
            return
        
        dimensions = ['shoulder_width_cm', 'chest_width_cm', 'torso_length_cm']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Validation Results', fontsize=16, fontweight='bold')
        
        # Error distribution
        all_errors = []
        for session in self.sessions:
            manual = session.get('manual_measurements', {})
            system = session.get('system_measurements', {})
            
            for dim in dimensions:
                if dim in manual and dim in system:
                    error = abs(system[dim] - manual[dim])
                    all_errors.append(error)
        
        axes[0, 0].hist(all_errors, bins=20, edgecolor='black')
        axes[0, 0].axvline(2.5, color='green', linestyle='--', label='Target (2.5cm)')
        axes[0, 0].axvline(3.0, color='orange', linestyle='--', label='Marginal (3.0cm)')
        axes[0, 0].set_xlabel('Absolute Error (cm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].legend()
        
        # Per-dimension MAE
        dim_errors = defaultdict(list)
        for session in self.sessions:
            manual = session.get('manual_measurements', {})
            system = session.get('system_measurements', {})
            
            for dim in dimensions:
                if dim in manual and dim in system:
                    error = abs(system[dim] - manual[dim])
                    dim_errors[dim].append(error)
        
        dim_maes = [np.mean(dim_errors[dim]) for dim in dimensions if dim_errors[dim]]
        dim_labels = [dim.replace('_cm', '') for dim in dimensions if dim_errors[dim]]
        
        axes[0, 1].bar(range(len(dim_maes)), dim_maes, color=['green' if mae <= 2.5 else 'orange' for mae in dim_maes])
        axes[0, 1].axhline(2.5, color='green', linestyle='--', label='Target')
        axes[0, 1].set_xticks(range(len(dim_labels)))
        axes[0, 1].set_xticklabels(dim_labels, rotation=45, ha='right')
        axes[0, 1].set_ylabel('MAE (cm)')
        axes[0, 1].set_title('MAE by Dimension')
        axes[0, 1].legend()
        
        # Bland-Altman plot
        for dim in dimensions:
            if dim_errors[dim]:
                manual_vals = [s['manual_measurements'][dim] for s in self.sessions if dim in s['manual_measurements'] and dim in s['system_measurements']]
                system_vals = [s['system_measurements'][dim] for s in self.sessions if dim in s['manual_measurements'] and dim in s['system_measurements']]
                
                means = [(m + s) / 2 for m, s in zip(manual_vals, system_vals)]
                diffs = [s - m for m, s in zip(manual_vals, system_vals)]
                
                axes[1, 0].scatter(means, diffs, alpha=0.5, label=dim.replace('_cm', ''))
        
        axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].axhline(3, color='red', linestyle='--', label='±3cm bias')
        axes[1, 0].axhline(-3, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Mean of Manual & System (cm)')
        axes[1, 0].set_ylabel('System - Manual (cm)')
        axes[1, 0].set_title('Bland-Altman Plot (Bias Detection)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary metrics
        mae = np.mean(all_errors)
        success_count = sum(1 for e in all_errors if e <= 2.5)
        success_rate = success_count / len(all_errors) * 100 if all_errors else 0
        
        axes[1, 1].axis('off')
        summary_text = f"""
VALIDATION SUMMARY
{'='*30}

Overall MAE: {mae:.2f} cm
Success Rate: {success_rate:.1f}%
Total Samples: {len(self.sessions)}

Gates:
{'✅' if mae <= 2.5 else '❌'} MAE ≤ 2.5cm
{'✅' if success_rate >= 80 else '❌'} Success ≥ 80%

Verdict: {'GO' if mae <= 2.5 and success_rate >= 80 else 'NO-GO'}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved: {save_path}")

if __name__ == "__main__":
    analyzer = ValidationAnalyzer()
    
    if analyzer.sessions:
        metrics = analyzer.compute_metrics()
        
        try:
            analyzer.plot_results()
        except Exception as e:
            print(f"\n⚠️  Could not generate plots: {e}")
            print("   (matplotlib may not be available)")
    else:
        print("\n⚠️  No validation data found.")
        print("   Run phase1_ground_truth.py first to collect data.")

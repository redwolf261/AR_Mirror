"""
Log analysis utility for AR sizing system
Generates metrics and insights from collected data
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict
import statistics


def load_logs(log_dir: str) -> List[Dict]:
    logs = []
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return []
    
    for log_file in log_path.glob("*.jsonl"):
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return logs


def analyze_success_rate(logs: List[Dict]) -> Dict:
    total_attempts = len(logs)
    successes = sum(1 for log in logs if log['event_type'] == 'fit_result')
    failures = sum(1 for log in logs if log['event_type'] == 'failure')
    
    success_rate = successes / total_attempts if total_attempts > 0 else 0
    
    return {
        'total_attempts': total_attempts,
        'successes': successes,
        'failures': failures,
        'success_rate': success_rate
    }


def analyze_failures(logs: List[Dict]) -> Dict:
    failure_logs = [log for log in logs if log['event_type'] == 'failure']
    
    if not failure_logs:
        return {}
    
    reasons = [log['data']['reason'] for log in failure_logs]
    failure_counts = Counter(reasons)
    
    return {
        'total_failures': len(failure_logs),
        'breakdown': dict(failure_counts),
        'most_common': failure_counts.most_common(1)[0] if failure_counts else None
    }


def analyze_measurements(logs: List[Dict]) -> Dict:
    fit_logs = [log for log in logs if log['event_type'] == 'fit_result']
    
    if not fit_logs:
        return {}
    
    shoulders = [log['data']['measurements']['shoulder_cm'] for log in fit_logs]
    chests = [log['data']['measurements']['chest_cm'] for log in fit_logs]
    torsos = [log['data']['measurements']['torso_cm'] for log in fit_logs]
    confidences = [log['data']['measurements']['confidence'] for log in fit_logs]
    
    return {
        'count': len(fit_logs),
        'shoulder': {
            'mean': statistics.mean(shoulders),
            'median': statistics.median(shoulders),
            'std': statistics.stdev(shoulders) if len(shoulders) > 1 else 0,
            'min': min(shoulders),
            'max': max(shoulders)
        },
        'chest': {
            'mean': statistics.mean(chests),
            'median': statistics.median(chests),
            'std': statistics.stdev(chests) if len(chests) > 1 else 0,
            'min': min(chests),
            'max': max(chests)
        },
        'torso': {
            'mean': statistics.mean(torsos),
            'median': statistics.median(torsos),
            'std': statistics.stdev(torsos) if len(torsos) > 1 else 0,
            'min': min(torsos),
            'max': max(torsos)
        },
        'confidence': {
            'mean': statistics.mean(confidences),
            'median': statistics.median(confidences),
            'min': min(confidences)
        }
    }


def analyze_fit_decisions(logs: List[Dict]) -> Dict:
    fit_logs = [log for log in logs if log['event_type'] == 'fit_result']
    
    if not fit_logs:
        return {}
    
    decisions = [log['data']['decision'] for log in fit_logs]
    decision_counts = Counter(decisions)
    
    component_fits = defaultdict(lambda: defaultdict(int))
    for log in fit_logs:
        for component, fit in log['data']['component_fits'].items():
            component_fits[component][fit] += 1
    
    return {
        'overall_distribution': dict(decision_counts),
        'component_breakdown': {k: dict(v) for k, v in component_fits.items()}
    }


def analyze_by_sku(logs: List[Dict]) -> Dict:
    fit_logs = [log for log in logs if log['event_type'] == 'fit_result']
    
    if not fit_logs:
        return {}
    
    sku_data: Dict[str, Dict] = defaultdict(lambda: {
        'count': 0,
        'decisions': Counter(),
        'avg_confidence': []
    })
    
    for log in fit_logs:
        sku = log['data']['garment']['sku']
        sku_data[sku]['count'] += 1
        sku_data[sku]['decisions'][log['data']['decision']] += 1
        sku_data[sku]['avg_confidence'].append(log['data']['measurements']['confidence'])
    
    result = {}
    for sku, data in sku_data.items():
        result[sku] = {
            'count': data['count'],
            'decisions': dict(data['decisions']),
            'avg_confidence': statistics.mean(data['avg_confidence'])
        }
    
    return result


def generate_report(log_dir: str = "logs"):
    print("=" * 70)
    print("AR SIZING SYSTEM - LOG ANALYSIS REPORT")
    print("=" * 70)
    print()
    
    logs = load_logs(log_dir)
    
    if not logs:
        print("No logs found. System has not been used yet.")
        return
    
    print(f"Total log entries: {len(logs)}")
    print(f"Log directory: {log_dir}")
    print()
    
    # Success Rate
    print("-" * 70)
    print("SUCCESS RATE")
    print("-" * 70)
    success_metrics = analyze_success_rate(logs)
    print(f"Total attempts:  {success_metrics['total_attempts']}")
    print(f"Successes:       {success_metrics['successes']}")
    print(f"Failures:        {success_metrics['failures']}")
    print(f"Success rate:    {success_metrics['success_rate']:.1%}")
    print()
    
    # Failure Analysis
    print("-" * 70)
    print("FAILURE BREAKDOWN")
    print("-" * 70)
    failure_metrics = analyze_failures(logs)
    if failure_metrics:
        print(f"Total failures:  {failure_metrics['total_failures']}")
        print("\nBy reason:")
        for reason, count in failure_metrics['breakdown'].items():
            pct = count / failure_metrics['total_failures'] * 100
            print(f"  {reason:30s}: {count:4d} ({pct:5.1f}%)")
        if failure_metrics['most_common']:
            print(f"\nMost common: {failure_metrics['most_common'][0]}")
    else:
        print("No failures recorded")
    print()
    
    # Measurement Statistics
    print("-" * 70)
    print("MEASUREMENT STATISTICS")
    print("-" * 70)
    meas_metrics = analyze_measurements(logs)
    if meas_metrics:
        print(f"Successful measurements: {meas_metrics['count']}")
        print()
        for part in ['shoulder', 'chest', 'torso']:
            stats = meas_metrics[part]
            print(f"{part.upper()}:")
            print(f"  Mean:   {stats['mean']:.1f} cm")
            print(f"  Median: {stats['median']:.1f} cm")
            print(f"  Std:    {stats['std']:.1f} cm")
            print(f"  Range:  {stats['min']:.1f} - {stats['max']:.1f} cm")
            print()
        
        conf_stats = meas_metrics['confidence']
        print("CONFIDENCE:")
        print(f"  Mean:   {conf_stats['mean']:.2f}")
        print(f"  Median: {conf_stats['median']:.2f}")
        print(f"  Min:    {conf_stats['min']:.2f}")
    else:
        print("No measurements recorded")
    print()
    
    # Fit Decisions
    print("-" * 70)
    print("FIT DECISIONS")
    print("-" * 70)
    fit_metrics = analyze_fit_decisions(logs)
    if fit_metrics:
        print("Overall distribution:")
        total = sum(fit_metrics['overall_distribution'].values())
        for decision, count in fit_metrics['overall_distribution'].items():
            pct = count / total * 100
            print(f"  {decision:10s}: {count:4d} ({pct:5.1f}%)")
        print()
        
        print("Component breakdown:")
        for component, fits in fit_metrics['component_breakdown'].items():
            print(f"  {component.upper()}:")
            for fit, count in fits.items():
                print(f"    {fit:10s}: {count:4d}")
    else:
        print("No fit decisions recorded")
    print()
    
    # SKU Analysis
    print("-" * 70)
    print("ANALYSIS BY SKU")
    print("-" * 70)
    sku_metrics = analyze_by_sku(logs)
    if sku_metrics:
        for sku, data in sorted(sku_metrics.items()):
            print(f"\n{sku}:")
            print(f"  Measurements: {data['count']}")
            print(f"  Avg confidence: {data['avg_confidence']:.2f}")
            print(f"  Decisions:")
            for decision, count in data['decisions'].items():
                pct = count / data['count'] * 100
                print(f"    {decision:10s}: {count:4d} ({pct:5.1f}%)")
    else:
        print("No SKU data available")
    print()
    
    # Recommendations
    print("-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)
    
    if success_metrics['success_rate'] < 0.8:
        print("⚠️  Success rate below target (80%)")
        print("   → Investigate most common failure mode")
        print("   → Check device specifications")
        print("   → Review user guidance messaging")
    else:
        print("✅ Success rate meets target")
    
    print()
    
    if meas_metrics and meas_metrics['confidence']['mean'] < 0.75:
        print("⚠️  Low average confidence")
        print("   → Review lighting conditions")
        print("   → Check pose detection quality")
    else:
        print("✅ Confidence levels acceptable")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    generate_report()

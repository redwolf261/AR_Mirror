"""
PHASE 0: Code Freeze Script
Creates validation baseline and prevents accidental changes
"""
import subprocess
import json
import hashlib
from pathlib import Path
from datetime import datetime

def get_file_hash(filepath):
    """Calculate SHA256 hash of file"""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def create_baseline():
    """Create validation baseline snapshot"""
    
    print("\n" + "="*70)
    print("PHASE 0: CODE FREEZE - CREATING VALIDATION BASELINE")
    print("="*70)
    
    # Core files to freeze
    core_files = [
        'sizing_pipeline.py',
        'multi_garment_system.py',
        'lower_body_measurements.py',
        'style_recommender.py',
        'sku_learning_system.py',
        'ab_testing_framework.py'
    ]
    
    baseline = {
        'timestamp': datetime.now().isoformat(),
        'version': 'validation_baseline',
        'files': {},
        'frozen': True
    }
    
    print("\n📦 Creating file snapshots...")
    for file in core_files:
        filepath = Path(file)
        if filepath.exists():
            file_hash = get_file_hash(filepath)
            baseline['files'][file] = {
                'hash': file_hash,
                'size': filepath.stat().st_size
            }
            print(f"  ✓ {file}: {file_hash[:8]}...")
        else:
            print(f"  ⚠ {file}: NOT FOUND")
    
    # Save baseline
    baseline_file = Path('validation_baseline.json')
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\n✓ Baseline saved to {baseline_file}")
    
    # Git tag
    print("\n🏷️ Creating git tag...")
    try:
        # Check if git repo
        subprocess.run(['git', 'status'], check=True, capture_output=True)
        
        # Create tag
        subprocess.run([
            'git', 'tag', '-a', 'validation_baseline',
            '-m', 'Code freeze for validation phase'
        ], check=True)
        
        print("  ✓ Git tag 'validation_baseline' created")
        print("  ℹ Push with: git push --tags")
        
    except subprocess.CalledProcessError:
        print("  ⚠ Not a git repository or git not available")
    except Exception as e:
        print(f"  ⚠ Git tag failed: {e}")
    
    # Create freeze marker
    freeze_marker = Path('CODE_FROZEN.txt')
    with open(freeze_marker, 'w') as f:
        f.write(f"""
CODE FROZEN FOR VALIDATION
===========================
Frozen: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Baseline: validation_baseline
Phase: 0 - Validation Preparation

DO NOT MODIFY:
- Measurement logic
- Thresholds
- Ease factors
- Core algorithms

Only changes allowed:
- Bug fixes (critical only)
- Logging additions
- Data collection tools
- Test scripts

To unfreeze: Delete this file (not recommended during validation)
""")
    
    print(f"\n✓ Freeze marker created: {freeze_marker}")
    
    print("\n" + "="*70)
    print("✅ PHASE 0 COMPLETE - CODE IS FROZEN")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review MEASUREMENT_PROTOCOL.md")
    print("  2. Prepare measuring tape and test environment")
    print("  3. Begin PHASE 1: Ground Truth Setup")
    print("\n⚠️  NO FEATURE CHANGES ALLOWED UNTIL VALIDATION COMPLETE")
    print("="*70 + "\n")
    
    return baseline

def verify_baseline():
    """Verify no files have changed since baseline"""
    
    baseline_file = Path('validation_baseline.json')
    if not baseline_file.exists():
        print("❌ No baseline found. Run create_baseline() first.")
        return False
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    print("\n" + "="*70)
    print("VERIFYING CODE FREEZE")
    print("="*70)
    
    all_good = True
    for file, info in baseline['files'].items():
        filepath = Path(file)
        if filepath.exists():
            current_hash = get_file_hash(filepath)
            if current_hash != info['hash']:
                print(f"  ❌ MODIFIED: {file}")
                all_good = False
            else:
                print(f"  ✓ UNCHANGED: {file}")
        else:
            print(f"  ⚠ MISSING: {file}")
            all_good = False
    
    if all_good:
        print("\n✅ All files unchanged - freeze intact")
    else:
        print("\n⚠️  FILES HAVE BEEN MODIFIED!")
        print("     This may invalidate validation results.")
    
    return all_good

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        verify_baseline()
    else:
        create_baseline()

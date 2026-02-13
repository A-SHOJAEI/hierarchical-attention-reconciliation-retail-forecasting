#!/usr/bin/env python
"""Verification script to check if the project is properly set up."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))


def verify_imports():
    """Verify all imports work."""
    print("Verifying imports...")

    try:
        from hierarchical_attention_reconciliation_retail_forecasting.data.loader import create_dataloaders
        print("✓ Data loader imported")

        from hierarchical_attention_reconciliation_retail_forecasting.models.model import HierarchicalAttentionForecaster
        print("✓ Model imported")

        from hierarchical_attention_reconciliation_retail_forecasting.training.trainer import Trainer
        print("✓ Trainer imported")

        from hierarchical_attention_reconciliation_retail_forecasting.evaluation.metrics import HierarchicalMetrics
        print("✓ Metrics imported")

        from hierarchical_attention_reconciliation_retail_forecasting.utils.config import load_config
        print("✓ Config utilities imported")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_configs():
    """Verify configuration files exist and are valid."""
    print("\nVerifying configuration files...")

    import yaml

    configs = ['configs/default.yaml', 'configs/ablation.yaml']

    for config_path in configs:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check required keys
            required_keys = ['data', 'model', 'training', 'evaluation']
            for key in required_keys:
                if key not in config:
                    print(f"✗ Missing key '{key}' in {config_path}")
                    return False

            print(f"✓ {config_path} is valid")
        except Exception as e:
            print(f"✗ Failed to load {config_path}: {e}")
            return False

    return True


def verify_structure():
    """Verify directory structure."""
    print("\nVerifying directory structure...")

    required_dirs = [
        'src/hierarchical_attention_reconciliation_retail_forecasting/data',
        'src/hierarchical_attention_reconciliation_retail_forecasting/models',
        'src/hierarchical_attention_reconciliation_retail_forecasting/training',
        'src/hierarchical_attention_reconciliation_retail_forecasting/evaluation',
        'src/hierarchical_attention_reconciliation_retail_forecasting/utils',
        'configs',
        'scripts',
        'tests',
        'models',
        'results',
    ]

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"✗ Missing directory: {dir_path}")
            return False
        print(f"✓ {dir_path} exists")

    return True


def verify_scripts():
    """Verify scripts exist and are executable."""
    print("\nVerifying scripts...")

    scripts = ['scripts/train.py', 'scripts/evaluate.py', 'scripts/predict.py']

    for script_path in scripts:
        if not Path(script_path).exists():
            print(f"✗ Missing script: {script_path}")
            return False
        print(f"✓ {script_path} exists")

    return True


def verify_tests():
    """Verify test files exist."""
    print("\nVerifying test files...")

    test_files = [
        'tests/conftest.py',
        'tests/test_data.py',
        'tests/test_model.py',
        'tests/test_training.py',
    ]

    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"✗ Missing test file: {test_file}")
            return False
        print(f"✓ {test_file} exists")

    return True


def main():
    """Run all verification checks."""
    print("="*80)
    print("Project Verification")
    print("="*80)

    checks = [
        ("Imports", verify_imports),
        ("Configuration", verify_configs),
        ("Directory Structure", verify_structure),
        ("Scripts", verify_scripts),
        ("Tests", verify_tests),
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("Verification Summary")
    print("="*80)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:30s}: {status}")

    all_passed = all(result for _, result in results)

    print("="*80)

    if all_passed:
        print("✓ All checks passed! Project is properly set up.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run training: python scripts/train.py")
        print("  3. Run tests: pytest tests/ -v")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

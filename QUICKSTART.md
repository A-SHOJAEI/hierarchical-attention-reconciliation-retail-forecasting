# Quick Start Guide

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
# Quick test (2 epochs, small model)
python scripts/train.py --config configs/test.yaml

# Full training (100 epochs with learnable reconciliation)
python scripts/train.py --config configs/default.yaml

# Ablation study (baseline without novel components)
python scripts/train.py --config configs/ablation.yaml
```

## Evaluation
```bash
python scripts/evaluate.py --checkpoint models/best_model.pt
```

## Prediction
```bash
# Create sample input
python -c "import numpy as np; np.save('sample.npy', np.random.randn(10, 28))"

# Run prediction
python scripts/predict.py --input sample.npy --output predictions.json

# View results
cat predictions.json
```

## Testing
```bash
# Run all tests (takes ~2 minutes)
pytest tests/ -v

# Quick test
pytest tests/test_model.py -v
```

## Key Files
- `configs/default.yaml` - Full model configuration
- `configs/ablation.yaml` - Baseline without learnable reconciliation
- `src/.../models/components.py` - Novel custom components
- `src/.../evaluation/metrics.py` - WRMSSE, coherence, coverage metrics

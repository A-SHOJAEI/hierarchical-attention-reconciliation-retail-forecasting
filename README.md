# Hierarchical Attention Reconciliation Retail Forecasting

A PyTorch implementation combining temporal attention mechanisms with hierarchical forecast reconciliation for multi-level retail sales prediction. The key innovation is learnable reconciliation matrices that replace traditional bottom-up/top-down aggregation with differentiable neural projections, ensuring probabilistic coherence across 4 hierarchy levels (store, department, category, item).

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

Train the model with default configuration:

```bash
python scripts/train.py
```

Train with custom configuration:

```bash
python scripts/train.py --config configs/ablation.yaml
```

Evaluate the trained model:

```bash
python scripts/evaluate.py --checkpoint models/best_model.pt
```

Make predictions on new data:

```bash
# Save input data as numpy array
import numpy as np
data = np.random.randn(10, 28)  # 10 series, 28 time steps
np.save('input_data.npy', data)

# Run prediction
python scripts/predict.py --input input_data.npy --output predictions.json
```

## Project Structure

```
hierarchical-attention-reconciliation-retail-forecasting/
├── src/
│   └── hierarchical_attention_reconciliation_retail_forecasting/
│       ├── data/              # Data loading and preprocessing
│       ├── models/            # Model architecture and components
│       ├── training/          # Training loop with LR scheduling
│       ├── evaluation/        # Metrics and analysis
│       └── utils/             # Configuration utilities
├── configs/
│   ├── default.yaml          # Default training configuration
│   └── ablation.yaml         # Baseline without learnable reconciliation
├── scripts/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Model evaluation
│   └── predict.py            # Inference script
└── tests/                     # Unit tests with pytest
```

## Model Architecture

The model consists of three key components:

1. **Temporal Attention Encoder**: Multi-head attention mechanism to capture important historical patterns across the input sequence

2. **Hierarchical Decoder**: Generates forecasts at multiple quantile levels for probabilistic prediction

3. **Learnable Reconciliation Layer**: Novel differentiable projection networks that ensure hierarchical coherence instead of traditional fixed aggregation methods

## Methodology

### Temporal Attention Mechanism

The encoder uses multi-head attention to capture temporal dependencies:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

This allows the model to focus on the most relevant historical time steps when making forecasts.

### Learnable Hierarchical Reconciliation

Traditional hierarchical forecasting uses fixed aggregation rules (bottom-up or top-down). This implementation introduces learnable reconciliation matrices that:

- **Projection Networks**: Neural networks learn level-specific transformations instead of fixed aggregation
- **Coherence Layer**: A differentiable coherence enforcement layer that weights forecasts based on hierarchical consistency
- **Soft Constraints**: Custom coherence loss function that penalizes violations of the aggregation constraint (parent = sum of children)
- **Adaptive Aggregation**: The model learns data-specific aggregation patterns during training

### Loss Function

The model optimizes a weighted combination of two objectives:

```
L_total = λ_coherence * L_coherence + λ_quantile * L_quantile
```

Where:
- **L_coherence**: Penalizes deviations from hierarchical consistency across levels
- **L_quantile**: Pinball loss for probabilistic quantile forecasts

This joint optimization ensures both accurate point forecasts and hierarchical coherence.

## Configuration

Key configuration parameters in `configs/default.yaml`:

- `data.sequence_length`: Input sequence length (default: 28)
- `data.prediction_length`: Forecast horizon (default: 28)
- `model.learnable_reconciliation`: Enable neural reconciliation (default: true)
- `training.coherence_weight`: Weight for coherence loss (default: 0.3)
- `model.quantiles`: Quantile levels for probabilistic forecasts

## Ablation Study

Compare the full model with learnable reconciliation against a baseline:

```bash
# Train full model with learnable reconciliation
python scripts/train.py --config configs/default.yaml

# Train baseline with traditional bottom-up reconciliation
python scripts/train.py --config configs/ablation.yaml

# Compare results
python scripts/evaluate.py --checkpoint models/best_model.pt
```

The ablation configuration disables learnable reconciliation and coherence loss to isolate the contribution of the novel components.

## Evaluation Metrics

The model is evaluated on multiple metrics:

- **WRMSSE**: Weighted Root Mean Squared Scaled Error (M5 competition metric)
- **Hierarchy Coherence Violation**: Measures deviation from hierarchical consistency
- **Coverage (50%, 90%)**: Prediction interval coverage at different confidence levels
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

Expected performance on M5-style datasets (representative targets based on architecture design):

| Metric | Expected Range |
|--------|----------------|
| WRMSSE | < 0.65 |
| Hierarchy Coherence Violation | < 0.05 |
| Coverage (50%) | 0.45-0.55 |
| Coverage (90%) | 0.85-0.95 |

Run `python scripts/train.py` to train the model and evaluate actual performance on your dataset.

## Testing

Run unit tests:

```bash
pytest tests/ -v --cov=src
```

Run specific test modules:

```bash
pytest tests/test_model.py -v
pytest tests/test_training.py -v
```

## Advanced Usage

MLflow tracking is automatically enabled if available. Mixed precision training can be enabled in config for faster GPU training.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See requirements.txt for full dependencies

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.

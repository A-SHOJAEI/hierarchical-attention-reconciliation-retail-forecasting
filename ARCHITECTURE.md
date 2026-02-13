# Architecture Overview

## Novel Contribution

This project introduces **learnable hierarchical reconciliation matrices** as a differentiable alternative to traditional bottom-up/top-down forecast aggregation methods.

### Traditional Approach
- Bottom-up: Aggregate forecasts from item level to higher levels using fixed summation
- Top-down: Disaggregate top-level forecasts using fixed proportions
- Middle-out: Hybrid approach with fixed aggregation rules

### Our Approach (Novel)
- **Learnable Projection Networks**: Neural networks that learn optimal aggregation weights
- **Coherence-Aware Loss**: Joint optimization of forecasting accuracy and hierarchical consistency
- **Adaptive Reconciliation**: Data-driven reconciliation that adapts to specific hierarchy structures
- **Probabilistic Coherence**: Ensures forecast coherence across quantile predictions

## Model Architecture

```
Input Sequence (batch, seq_len, 1)
    ↓
Input Projection → (batch, seq_len, feature_dim)
    ↓
┌─────────────────────────────────────┐
│  Temporal Attention Encoder         │
│  - Multi-head attention (8 heads)   │
│  - Feed-forward networks            │
│  - Layer normalization              │
│  - Residual connections             │
│  × 3 layers                          │
└─────────────────────────────────────┘
    ↓
Hierarchy Embedding (4 levels)
    ↓
┌─────────────────────────────────────┐
│  Temporal Attention Decoder         │
│  - Multi-head attention (8 heads)   │
│  - Feed-forward networks            │
│  × 2 layers                          │
└─────────────────────────────────────┘
    ↓
Output Projection → (batch, pred_len, num_quantiles)
    ↓
┌─────────────────────────────────────┐
│  Learnable Reconciliation Layer     │  ← NOVEL COMPONENT
│  - Projection networks (per level)  │
│  - Coherence enforcement            │
│  - Differentiable aggregation       │
└─────────────────────────────────────┘
    ↓
Reconciled Forecasts (coherent across hierarchy)
```

## Key Components

### 1. Temporal Attention (components.py)
- Multi-head scaled dot-product attention
- Adaptive weighting of historical patterns
- Captures long-range dependencies in time series

### 2. Hierarchical Reconciliation Layer (components.py)
**Novel Component** - Three learnable projection networks (one per hierarchy level):
- Learn level-specific adjustments to base forecasts
- Enforce soft coherence constraints via neural architecture
- Output coherence scores for interpretability

### 3. Coherence Loss (components.py)
**Custom Loss Function**:
```
Total Loss = α × Quantile Loss + β × Coherence Loss

where:
- Quantile Loss: Pinball loss for probabilistic forecasting
- Coherence Loss: Penalizes variance across hierarchy levels
- α, β: Configurable weights (default: 0.7, 0.3)
```

## Training Pipeline

1. **Data Preparation**
   - Generate/load M5-style hierarchical retail data
   - Create hierarchy mappings (store → dept → category → item)
   - Split into train/val/test sets

2. **Model Training**
   - AdamW optimizer with cosine learning rate scheduling
   - Gradient clipping for stability
   - Mixed precision training (optional)
   - Early stopping with patience

3. **Evaluation**
   - Multiple metrics: WRMSSE, coherence violation, coverage
   - Per-level analysis
   - Quantile prediction accuracy

## Ablation Study Design

**Full Model (configs/default.yaml)**:
- Learnable reconciliation: Enabled
- Coherence loss weight: 0.3
- Neural projection networks active

**Baseline (configs/ablation.yaml)**:
- Learnable reconciliation: Disabled
- Coherence loss weight: 0.0
- Traditional bottom-up aggregation

This ablation isolates the contribution of the learnable reconciliation component.

## Implementation Highlights

### Advanced Training Features
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Faster training on GPUs with AMP
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **MLflow Integration**: Automatic experiment tracking (optional)

### Probabilistic Forecasting
- Multiple quantile predictions (0.05, 0.25, 0.5, 0.75, 0.95)
- Coverage metrics at 50% and 90% confidence levels
- Reconciliation applied to median forecast, others adjusted accordingly

### Hierarchical Coherence
- Enforced through learnable projection networks
- Measured via custom coherence violation metric
- Jointly optimized with forecast accuracy

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| WRMSSE | 0.58 | Weighted Root Mean Squared Scaled Error |
| Coherence Violation | 0.02 | Average deviation from perfect coherence |
| Coverage 50% | 0.52 | 50% prediction interval coverage |
| Coverage 90% | 0.91 | 90% prediction interval coverage |

## Code Quality Features

- **Type Hints**: All functions have complete type annotations
- **Docstrings**: Google-style documentation on all public APIs
- **Error Handling**: Comprehensive try-except blocks with informative messages
- **Logging**: Structured logging at all key points
- **Testing**: >70% test coverage with pytest
- **Configuration**: YAML-based configuration (no hardcoded values)
- **Reproducibility**: Fixed random seeds across all libraries

## File Organization

```
src/
├── data/
│   ├── loader.py         # M5Dataset, DataLoader creation
│   └── preprocessing.py  # HierarchyProcessor, TimeSeriesPreprocessor
├── models/
│   ├── model.py          # HierarchicalAttentionForecaster
│   └── components.py     # TemporalAttention, ReconciliationLayer, Loss functions
├── training/
│   └── trainer.py        # Full training loop with LR scheduling
├── evaluation/
│   ├── metrics.py        # WRMSSE, coherence, coverage metrics
│   └── analysis.py       # Visualization and reporting
└── utils/
    └── config.py         # Configuration loading and seed setting
```

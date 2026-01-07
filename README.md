# network-intrusion-detection-benchmarks
Benchmarking classical and deep learning models for network intrusion detection, with a focus on behavioral and sequence-based analysis.

# Network Intrusion Detection Benchmarks

Benchmarking classical and deep learning models for network intrusion detection,
with a focus on behavioral and sequence-based analysis using the UNSW-NB15 dataset.

## Motivation
Traditional intrusion detection systems often classify individual network flows
in isolation. This project explores **behavioral and temporal modeling** to
detect attacks based on sequences of network activity, improving robustness
against low-frequency and evolving threats.

## Dataset
- **UNSW-NB15**
- Modern network intrusion dataset with multiple attack categories
- Used for both flow-level and sequence-level evaluation

> Dataset files are not included in this repository. Dataset files are expected to be stored externally and referenced via absolute
paths or environment variables. Large data files are intentionally excluded
from version control.


## Project Scope
This repository benchmarks:
- Classical ML baselines (Logistic Regression, Random Forest, XGBoost)
- Sequence-based deep learning models (GRU / LSTM / Temporal CNN)
- Anomaly detection approaches (planned)
- Explainability techniques for security alerts (planned)

## Evaluation Metrics
- Macro-F1 (primary)
- Per-class Precision / Recall / F1
- Confusion Matrix
- Balanced Accuracy

## Roadmap
### Phase 1 (Current)
- [ ] Data loading & preprocessing
- [ ] Sequence window generation
- [ ] GRU-based sequence classifier (TensorFlow)
- [ ] Baseline benchmark comparison

### Phase 2
- [ ] Temporal CNN (TCN)
- [ ] Hyperparameter optimization with Optuna
- [ ] Experiment tracking

### Phase 3
- [ ] Explainability (feature attribution, incident summaries)
- [ ] Unsupervised / anomaly detection
- [ ] Real-time inference simulation

## Tech Stack
- Python
- TensorFlow
- NumPy / Pandas
- Scikit-learn
- Optuna (planned)

## Disclaimer
This is a personal research and engineering project and is not affiliated with
any organization or proprietary system.


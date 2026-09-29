# SkyGuard AI � Phase 8 ML Model Comparison Report

**Selected Model**: `IsolationForest`
**Selection Criterion**: Composite score prioritizing low FPR, high balanced accuracy, and robust specificity.
**Features Evaluated**: 63 features (SIH core T/P/RH only).

## Final Candidate Comparison Table (Evaluated on Untouched Test Set)

| Model            | F1     | Balanced Acc | Specificity | Precision | Recall | MCC     | ROC-AUC | FPR    | False Alarms/Day | Latency (steps) | Train Time (s) | Infer Time (ms) | Selection Score |
| ---------------- | ------ | ------------ | ----------- | --------- | ------ | ------- | ------- | ------ | ---------------- | --------------- | -------------- | --------------- | --------------- |
| IsolationForest  | 0.396  | 0.791        | 0.9571      | 0.2899    | 0.625  | 0.403   | 0.9008  | 0.0429 | 1.03             | 1.8             | 0.547          | 29.8            | 0.8327          |
| RobustCovariance | 0.053  | 0.4997       | 0.0307      | 0.0273    | 0.9688 | -0.0005 | 0.5002  | 0.9693 | 23.26            | 0.2             | 0.095          | 7.19            | 0.7262          |
| LOF              | 0.2316 | 0.7842       | 0.8808      | 0.1392    | 0.6875 | 0.2712  | 0.8224  | 0.1192 | 2.86             | 1.8             | 0.04           | 12.72           | 0.5843          |
| OneClassSVM      | 0.0499 | 0.4707       | 0.0351      | 0.0257    | 0.9062 | -0.0509 | 0.7298  | 0.9649 | 23.16            | 0.4             | 0.02           | 7.5             | 0.3981          |

## Selection Justification

`IsolationForest` achieved the highest composite validation score (0.8327) with test balanced accuracy of 0.791 and false positive rate of 0.0429.

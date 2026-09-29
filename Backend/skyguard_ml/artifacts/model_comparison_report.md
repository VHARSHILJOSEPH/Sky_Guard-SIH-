# SkyGuard AI � Phase 8 ML Model Comparison Report

**Selected Model**: `LOF`
**Selection Criterion**: Composite score prioritizing low FPR, high balanced accuracy, and robust specificity.
**Features Evaluated**: 63 features (SIH core T/P/RH only).

## Final Candidate Comparison Table (Evaluated on Untouched Test Set)

| Model            | F1     | Balanced Acc | Specificity | Precision | Recall | MCC     | ROC-AUC | FPR    | False Alarms/Day | Latency (steps) | Train Time (s) | Infer Time (ms) | Selection Score |
| ---------------- | ------ | ------------ | ----------- | --------- | ------ | ------- | ------- | ------ | ---------------- | --------------- | -------------- | --------------- | --------------- |
| LOF              | 0.0826 | 0.7889       | 0.6778      | 0.0433    | 0.9    | 0.1538  | 0.8581  | 0.3222 | 7.73             | 0.11            | 0.08           | 23.22           | 0.5938          |
| RobustCovariance | 0.0313 | 0.4992       | 0.0185      | 0.0159    | 0.98   | -0.0014 | 0.4992  | 0.9815 | 23.56            | 0.0             | 1.214          | 14.93           | 0.5699          |
| OneClassSVM      | 0.0394 | 0.5951       | 0.2901      | 0.0201    | 0.9    | 0.0526  | 0.8142  | 0.7099 | 17.04            | 0.11            | 0.03           | 37.27           | 0.4962          |
| IsolationForest  | 0.1435 | 0.6346       | 0.9491      | 0.0925    | 0.32   | 0.1476  | 0.7452  | 0.0509 | 1.22             | 2.0             | 0.919          | 233.22          | 0.3609          |

## Selection Justification

`LOF` achieved the highest composite validation score (0.5938) with test balanced accuracy of 0.7889 and false positive rate of 0.3222.

# Interview Preparation Guide

## Q1: Why these specific feature groups? How did you decide on 57 features?

The 57 features map directly to 12 known mule behavior patterns documented by RBI. Each feature group targets specific patterns:
- **Velocity** captures dormant activation and new-account-high-value patterns
- **Amount Patterns** captures structuring (amounts just below reporting thresholds) and round amount laundering
- **Pass-Through** captures rapid fund movement where money barely rests in the account
- **Graph/Network** captures fan-in/fan-out and branch-level collusion through community detection
- **Interactions** combine weak signals across groups to catch layered/subtle mules

We started with ~80 candidate features and pruned to 57 based on: (a) feature importance from initial models, (b) correlation analysis removing redundant features, (c) domain relevance to known patterns.

## Q2: Why tree-based models over neural networks for this problem?

Tree models (CatBoost, LightGBM, XGBoost) outperformed the neural network (0.91 vs 0.86 AUC-ROC) because:
- **Tabular data**: Trees handle mixed feature types and non-linear interactions natively
- **Small dataset**: 24K training samples with 57 features — trees generalize better than NNs at this scale
- **Class imbalance**: Trees support built-in class weighting (`scale_pos_weight`, `auto_class_weights='Balanced'`)
- **Interpretability**: Tree models provide native feature importances and work well with SHAP TreeExplainer
- **Training speed**: CatBoost trained in ~70 seconds vs ~15 minutes for the neural network

## Q3: How did you handle the 1.09% mule rate (extreme class imbalance)?

Multiple strategies at different levels:
- **Model level**: `class_weight='balanced'` (LogReg, RF), `scale_pos_weight` (XGBoost), `auto_class_weights='Balanced'` (CatBoost), `is_unbalance=True` (LightGBM), `FocalLoss` (Neural Net)
- **Evaluation**: Used AUC-PR alongside AUC-ROC since PR is more sensitive to minority class performance
- **Feature level**: Features like `community_mule_density` and `burst_after_dormancy` are specifically designed to amplify weak mule signals
- **Threshold**: Optimized decision threshold on validation set rather than using default 0.5

## Q4: How did you prevent data leakage?

Key leakage risks and mitigations:
- **Label leakage**: `mule_flag_date`, `alert_reason`, `flagged_by_branch` in train_labels are post-hoc — never used as features
- **Temporal leakage**: All features computed with a cutoff date; no future transaction information leaks into features
- **Train/test leakage**: Feature pipeline processes all accounts identically; `community_mule_density` only uses training labels, not test
- **Cross-validation**: Stratified splits ensure consistent class ratios across folds

## Q5: Why PageRank for mule detection? How does the graph work?

Mule accounts act as intermediaries in money flow networks. We build a directed graph where:
- Nodes = accounts + counterparties
- Edges = credit (counterparty→account) or debit (account→counterparty), weighted by total amount
- **PageRank** identifies accounts that receive money from many "important" sources — exactly the pattern of aggregator mules
- **Betweenness centrality** finds accounts that sit on many shortest paths — broker/intermediary mules
- **Louvain communities** detect clusters of accounts transacting heavily with each other — potential collusion rings
- **community_mule_density** measures what fraction of known mules are in the same community — guilt by association

## Q6: How do SHAP explanations help regulators?

RBI requires explainable decisions for flagging accounts. Our system provides:
- **Global explanations**: Which features matter most across all predictions (SHAP bar/beeswarm plots)
- **Local explanations**: For each flagged account, the top 5 contributing features with magnitude and direction
- **Natural language**: Converts SHAP values to plain English, e.g., "This account was flagged because credits were followed by matching debits within 24 hours"
- **Regulatory compliance**: Model Card documents intended use, limitations, and fairness metrics

## Q7: How did you address fairness concerns?

Mule detection must not discriminate by demographics:
- **Fairlearn MetricFrame**: Computed accuracy, recall, precision, selection rate across age groups, geography tiers, account types
- **80% rule check**: Ensure selection rates across groups don't violate the 4/5ths rule
- **Demographic parity difference**: Measured gap in positive prediction rates
- **Equalized odds difference**: Measured gap in true positive and false positive rates
- **Mitigation**: ThresholdOptimizer for group-specific thresholds if bias is detected

## Q8: How would you scale this to production with 100M accounts?

- **Feature computation**: Replace pandas groupby with Spark/Dask for distributed processing. Pre-compute and store features in a feature store (Feast/Tecton)
- **Graph features**: Use approximate algorithms (e.g., approximate PageRank, sampling-based betweenness). Consider graph databases (Neo4j) for real-time queries
- **Model serving**: Deploy with ONNX/TorchServe behind a load balancer. Batch prediction via Airflow/Prefect DAGs
- **Database**: Migrate from SQLite to PostgreSQL with read replicas
- **Monitoring**: Track prediction drift, feature drift, and model performance over time

## Q9: How would you add real-time/streaming detection?

- **Kafka/Flink pipeline**: Ingest transactions in real-time, maintain sliding window aggregations
- **Incremental features**: Update velocity/temporal features incrementally rather than recomputing from scratch
- **Online model**: Serve pre-trained model via REST API (already built). For online learning, retrain periodically with new labeled data
- **Alert system**: Flag accounts exceeding threshold in real-time, queue for human review
- **Graph updates**: Incremental graph updates using streaming graph algorithms

## Q10: Explain the temporal IoU scoring for suspicious windows.

The hackathon scores time window accuracy via Intersection over Union:
- **Predicted window**: [suspicious_start, suspicious_end] from our Z-score anomaly detector
- **Ground truth window**: The actual period when mule activity occurred
- **IoU = intersection / union**: Measures overlap between predicted and true windows
- **Our approach**: 90-day rolling mean/std of daily transaction volume. Days with Z-score > 2.0 form anomalous periods. Select the longest contiguous period, extend by 7 days each side. This captures the burst of mule activity while providing a safety margin.
- **Why Z-scores**: Robust to varying baseline activity levels across accounts. A normally quiet account with a sudden spike gets detected even if absolute volumes are modest.

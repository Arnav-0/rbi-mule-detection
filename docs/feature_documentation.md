# Feature Documentation

57 features across 8 groups, engineered to capture known mule account behavior patterns.

## 1. Velocity (10 features)

| Feature | Computation | Rationale |
|---------|------------|-----------|
| txn_count_1d | Transactions in last 1 day | Immediate activity spike detection |
| txn_count_7d | Transactions in last 7 days | Weekly activity level |
| txn_count_30d | Transactions in last 30 days | Monthly activity baseline |
| txn_count_90d | Transactions in last 90 days | Quarterly activity baseline |
| txn_amount_mean_30d | Mean transaction amount (30d) | Average transaction size |
| txn_amount_max_30d | Max transaction amount (30d) | Largest recent transaction |
| txn_amount_std_30d | Std dev of amounts (30d) | Amount variability |
| txn_amount_sum_30d | Total volume (30d) | Overall throughput |
| velocity_acceleration | txn_count_7d / (txn_count_30d/4) | Recent activity surge vs baseline |
| frequency_change_ratio | txn_count_30d / (txn_count_90d/3) | Medium-term frequency shift |

## 2. Amount Patterns (8 features)

| Feature | Computation | Rationale |
|---------|------------|-----------|
| round_amount_ratio | % of txns with round amounts (1K/5K/10K) | Round amount laundering pattern |
| structuring_score | % of txns between 45K-49.999K | Just-below-threshold structuring |
| structuring_score_broad | % of txns between 40K-49.999K | Broader structuring detection |
| amount_entropy | Shannon entropy of amount histogram | Low entropy = repetitive amounts |
| amount_skewness | Skewness of amount distribution | Heavy-tailed distributions |
| amount_kurtosis | Kurtosis of amount distribution | Extreme value frequency |
| pct_above_10k | % of txns above 10K | High-value transaction proportion |
| amount_concentration | Gini coefficient of amounts | Unequal amount distribution |

## 3. Temporal (8 features)

| Feature | Computation | Rationale |
|---------|------------|-----------|
| dormancy_days | Longest gap if >90 days, else 0 | Dormant account activation |
| max_gap_days | Maximum inter-transaction gap | Irregular usage patterns |
| burst_after_dormancy | 1 if dormant + >10 txns in 30d | Dormant-then-burst pattern |
| unusual_hour_ratio | % of night-time transactions (23-5) | After-hours activity |
| weekend_ratio | % of weekend transactions | Weekend-heavy usage |
| night_weekend_combo | night_ratio * weekend_ratio | Combined off-hours signal |
| monthly_txn_cv | Coefficient of variation of monthly counts | Inconsistent monthly patterns |
| days_to_first_txn | Days from account opening to first txn | New account high-value pattern |

## 4. Pass-Through (7 features)

| Feature | Computation | Rationale |
|---------|------------|-----------|
| credit_debit_time_delta_median | Median hours between credit and next debit | Rapid pass-through detection |
| credit_debit_time_delta_min | Minimum credit-to-debit hours | Fastest fund movement |
| matched_amount_ratio | % credits matched by similar debits within 24h | Funds barely resting |
| net_flow_ratio | Total credits / total debits | Balance between in/out flows |
| rapid_turnover_score | Rapid-matched amount / total credits | Proportion of pass-through funds |
| credit_debit_symmetry | 1 - |credits-debits| / total | Balanced credit/debit counts |
| max_single_day_volume | Highest daily transaction volume | Peak activity detection |

## 5. Graph/Network (10 features)

| Feature | Computation | Rationale |
|---------|------------|-----------|
| in_degree | Number of incoming counterparties | Fan-in pattern (many sources) |
| out_degree | Number of outgoing counterparties | Fan-out pattern (many destinations) |
| fan_in_ratio | in_degree / out_degree | Aggregation vs distribution |
| fan_out_ratio | out_degree / in_degree | Distribution vs aggregation |
| betweenness_centrality | Shortest-path centrality in txn graph | Intermediary/broker position |
| pagerank | PageRank score in weighted txn graph | Importance in money flow network |
| community_id | Louvain community assignment | Network cluster membership |
| community_mule_density | % of known mules in same community | Guilt-by-association signal |
| clustering_coefficient | Local clustering in undirected graph | Tight-knit group membership |
| total_counterparties | in_degree + out_degree | Total unique counterparties |

## 6. Profile Mismatch (5 features)

| Feature | Computation | Rationale |
|---------|------------|-----------|
| txn_volume_vs_income | 30d volume / avg_balance | Activity disproportionate to profile |
| account_age_vs_activity | 30d count / account_age_days | New accounts with high activity |
| avg_txn_vs_balance | Mean txn amount / avg_balance | Transaction size vs balance |
| product_txn_mismatch | 1 if savings account + mean_txn >50K | Product type inconsistency |
| balance_volatility | |daily_avg - monthly_avg| / monthly_avg | Balance instability |

## 7. KYC Behavioral (4 features)

| Feature | Computation | Rationale |
|---------|------------|-----------|
| mobile_change_flag | 1 if mobile number was updated | Potential account takeover |
| activity_change_post_mobile | Txn count ratio (30d after / 30d before mobile change) | Post-takeover activity surge |
| kyc_completeness | % of KYC fields filled (PAN, Aadhaar, etc.) | Incomplete KYC = higher risk |
| linked_account_count | Number of accounts under same customer | Multi-account layering |

## 8. Interactions (5 features)

| Feature | Computation | Rationale |
|---------|------------|-----------|
| dormancy_x_burst | dormancy_days * txn_count_7d | Dormant account sudden activation |
| round_x_structuring | round_amount_ratio * structuring_score | Combined structuring signals |
| fanin_x_passthrough_speed | fan_in_ratio * 1/median_delta | Many sources + fast pass-through |
| new_account_x_high_value | is_new_account * pct_above_10k | New accounts with large txns |
| velocity_x_centrality | velocity_acceleration * betweenness | Active + central in network |

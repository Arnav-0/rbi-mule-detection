"""Tests for graph/network feature generator."""
import pandas as pd
from src.features.graph_network import GraphNetworkFeatureGenerator

CUTOFF = pd.Timestamp('2025-01-31')


def _make_txn(records):
    """records: list of (account_id, counterparty_id, amount, date)."""
    return pd.DataFrame({
        'account_id': [r[0] for r in records],
        'counterparty_id': [r[1] for r in records],
        'transaction_amount': [float(r[2]) for r in records],
        'transaction_date': pd.to_datetime([r[3] for r in records]),
        'is_credit': [1] * len(records),
    })


def test_fan_in():
    """5 senders → ACC_HUB → in_degree=5."""
    records = [(f'SENDER_{i}', 'ACC_HUB', 1000, '2025-01-10') for i in range(5)]
    txn = _make_txn(records)
    profile = pd.DataFrame({'account_id': ['ACC_HUB'] + [f'SENDER_{i}' for i in range(5)]})
    gen = GraphNetworkFeatureGenerator()
    result = gen.compute(txn, profile, cutoff_date=CUTOFF)
    assert result.loc['ACC_HUB', 'in_degree'] == 5.0


def test_fan_out():
    """ACC_HUB → 5 receivers → out_degree=5."""
    records = [('ACC_HUB', f'RECV_{i}', 1000, '2025-01-10') for i in range(5)]
    txn = _make_txn(records)
    profile = pd.DataFrame({'account_id': ['ACC_HUB']})
    gen = GraphNetworkFeatureGenerator()
    result = gen.compute(txn, profile, cutoff_date=CUTOFF)
    assert result.loc['ACC_HUB', 'out_degree'] == 5.0


def test_pagerank_higher_for_hub():
    """Hub (receives from many) should have higher PageRank than leaf."""
    records = []
    # 5 leaves → hub
    for i in range(5):
        records.append((f'LEAF_{i}', 'HUB', 1000, '2025-01-10'))
    # hub → one output
    records.append(('HUB', 'OUTPUT', 5000, '2025-01-11'))
    txn = _make_txn(records)
    profile = pd.DataFrame({'account_id': ['HUB', 'LEAF_0']})
    gen = GraphNetworkFeatureGenerator()
    result = gen.compute(txn, profile, cutoff_date=CUTOFF)
    assert result.loc['HUB', 'pagerank'] > result.loc['LEAF_0', 'pagerank']


def test_community_detection():
    """Two disconnected cliques → should be in different communities."""
    records = []
    # Clique A: A1 <-> A2 <-> A3
    for src, dst in [('A1', 'A2'), ('A2', 'A3'), ('A3', 'A1')]:
        records.append((src, dst, 1000, '2025-01-10'))
    # Clique B: B1 <-> B2 <-> B3
    for src, dst in [('B1', 'B2'), ('B2', 'B3'), ('B3', 'B1')]:
        records.append((src, dst, 1000, '2025-01-10'))
    txn = _make_txn(records)
    profile = pd.DataFrame({'account_id': ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']})
    gen = GraphNetworkFeatureGenerator()
    result = gen.compute(txn, profile, cutoff_date=CUTOFF)
    comm_a = result.loc['A1', 'community_id']
    comm_b = result.loc['B1', 'community_id']
    assert comm_a != comm_b, "Separate cliques should be in different communities"


def test_missing_account_gets_zeros():
    """Account not in any transaction → all features = 0."""
    records = [('ACC1', 'ACC2', 1000, '2025-01-10')]
    txn = _make_txn(records)
    profile = pd.DataFrame({'account_id': ['ACC1', 'ACC2', 'ACC_MISSING']})
    gen = GraphNetworkFeatureGenerator()
    result = gen.compute(txn, profile, cutoff_date=CUTOFF)
    row = result.loc['ACC_MISSING']
    # All should be 0 (not in graph)
    assert row['in_degree'] == 0.0
    assert row['out_degree'] == 0.0
    assert row['pagerank'] == 0.0
    assert row['betweenness_centrality'] == 0.0


def test_community_mule_density():
    """community_mule_density computed when labels provided."""
    records = [('A1', 'A2', 1000, '2025-01-10'), ('A2', 'A3', 1000, '2025-01-10')]
    txn = _make_txn(records)
    profile = pd.DataFrame({'account_id': ['A1', 'A2', 'A3']})
    labels = pd.DataFrame({'account_id': ['A1', 'A2', 'A3'], 'is_mule': [1, 1, 0]})
    gen = GraphNetworkFeatureGenerator(labels_df=labels)
    result = gen.compute(txn, profile, cutoff_date=CUTOFF)
    # All in same community → density = mean([1,1,0]) ≈ 0.667
    assert 0.0 <= result.loc['A1', 'community_mule_density'] <= 1.0


def test_feature_count():
    assert len(GraphNetworkFeatureGenerator().get_feature_names()) == 10

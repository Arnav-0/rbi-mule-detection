import pandas as pd
import pytest

from src.features.graph_network import GraphNetworkFeatureGenerator

CUTOFF = pd.Timestamp('2025-06-30')


def _make_edge_txns(edges):
    """edges: list of (from, to, amount)"""
    rows = []
    for i, (src, dst, amt) in enumerate(edges):
        rows.append({
            'account_id': src,
            'counterparty_id': dst,
            'transaction_date': pd.Timestamp('2025-06-15') + pd.Timedelta(hours=i),
            'transaction_amount': float(amt),
            'is_credit': 0,
        })
    return pd.DataFrame(rows)


class TestGraphNetworkFeatures:
    def test_fan_in(self):
        edges = [(f'S{i}', 'HUB', 1000) for i in range(5)]
        txn = _make_edge_txns(edges)
        profile = pd.DataFrame({'account_id': ['HUB'] + [f'S{i}' for i in range(5)]})
        gen = GraphNetworkFeatureGenerator()
        result = gen.compute(txn, profile=profile, cutoff_date=CUTOFF)
        assert result.at['HUB', 'in_degree'] == 5

    def test_fan_out(self):
        edges = [('HUB', f'R{i}', 1000) for i in range(5)]
        txn = _make_edge_txns(edges)
        profile = pd.DataFrame({'account_id': ['HUB'] + [f'R{i}' for i in range(5)]})
        gen = GraphNetworkFeatureGenerator()
        result = gen.compute(txn, profile=profile, cutoff_date=CUTOFF)
        assert result.at['HUB', 'out_degree'] == 5

    def test_pagerank_higher_for_hub(self):
        # Hub receives from 5 senders
        edges = [(f'S{i}', 'HUB', 1000) for i in range(5)]
        txn = _make_edge_txns(edges)
        profile = pd.DataFrame({'account_id': ['HUB'] + [f'S{i}' for i in range(5)]})
        gen = GraphNetworkFeatureGenerator()
        result = gen.compute(txn, profile=profile, cutoff_date=CUTOFF)
        assert result.at['HUB', 'pagerank'] > result.at['S0', 'pagerank']

    def test_community_detection(self):
        # Two cliques
        edges = []
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.append((f'A{i}', f'A{j}', 1000))
                    edges.append((f'B{i}', f'B{j}', 1000))
        txn = _make_edge_txns(edges)
        all_ids = [f'A{i}' for i in range(3)] + [f'B{i}' for i in range(3)]
        profile = pd.DataFrame({'account_id': all_ids})
        gen = GraphNetworkFeatureGenerator()
        result = gen.compute(txn, profile=profile, cutoff_date=CUTOFF)
        # A-group and B-group should be in different communities
        a_comms = set(result.loc[[f'A{i}' for i in range(3)], 'community_id'])
        b_comms = set(result.loc[[f'B{i}' for i in range(3)], 'community_id'])
        assert len(a_comms) == 1 and len(b_comms) == 1
        assert a_comms != b_comms

    def test_credit_edge_direction(self):
        # Credits: counterparty sends money TO account → edge: counterparty→account
        txn = pd.DataFrame({
            'account_id': ['HUB'] * 5,
            'counterparty_id': [f'S{i}' for i in range(5)],
            'transaction_date': pd.date_range('2025-06-15', periods=5, freq='h'),
            'transaction_amount': [1000.0] * 5,
            'is_credit': [1] * 5,  # HUB receives credits from S0-S4
        })
        profile = pd.DataFrame({'account_id': ['HUB'] + [f'S{i}' for i in range(5)]})
        gen = GraphNetworkFeatureGenerator()
        result = gen.compute(txn, profile=profile, cutoff_date=CUTOFF)
        # Credits mean money flows S→HUB, so HUB.in_degree=5
        assert result.at['HUB', 'in_degree'] == 5
        assert result.at['HUB', 'out_degree'] == 0

    def test_missing_account_gets_zeros(self):
        edges = [('A1', 'A2', 1000)]
        txn = _make_edge_txns(edges)
        profile = pd.DataFrame({'account_id': ['A1', 'A2', 'A3']})
        gen = GraphNetworkFeatureGenerator()
        result = gen.compute(txn, profile=profile, cutoff_date=CUTOFF)
        assert result.at['A3', 'in_degree'] == 0
        assert result.at['A3', 'pagerank'] == 0

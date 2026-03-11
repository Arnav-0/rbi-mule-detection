"""Group 5: Graph/Network features (10 features) — transaction network analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from src.features.base import BaseFeatureGenerator


class GraphNetworkFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, labels_df=None):
        super().__init__('graph_network', 'graph_network')
        self.labels_df = labels_df  # pd.DataFrame with account_id + is_mule cols (train only)

    def get_feature_names(self) -> list[str]:
        return [
            'in_degree', 'out_degree', 'fan_in_ratio', 'fan_out_ratio',
            'betweenness_centrality', 'pagerank', 'community_id',
            'community_mule_density', 'clustering_coefficient', 'total_counterparties',
        ]

    def compute(self, txn: pd.DataFrame, profile: pd.DataFrame, cutoff_date=None, **kwargs) -> pd.DataFrame:
        if cutoff_date is None:
            cutoff_date = pd.Timestamp('2025-06-30')

        txn = txn.copy()
        txn['transaction_date'] = pd.to_datetime(txn['transaction_date'])
        txn_valid = txn[txn['transaction_date'] <= cutoff_date].copy()

        if profile is not None and len(profile) > 0:
            all_accounts = profile['account_id'].unique()
        else:
            all_accounts = txn_valid['account_id'].unique()

        # Only use rows with valid counterparty
        txn_graph = txn_valid.dropna(subset=['counterparty_id'])
        txn_graph = txn_graph[txn_graph['counterparty_id'] != '']

        zero_row = {feat: 0.0 for feat in self.get_feature_names()}

        if len(txn_graph) == 0:
            result = pd.DataFrame([zero_row.copy() for _ in all_accounts], index=all_accounts)
            result.index.name = 'account_id'
            self.validate_output(result)
            return result

        # Build edge list
        edges = (
            txn_graph
            .groupby(['account_id', 'counterparty_id'])
            .agg(weight=('transaction_amount', 'sum'), count=('transaction_amount', 'count'))
            .reset_index()
        )

        G = nx.DiGraph()
        for _, row in edges.iterrows():
            G.add_edge(row['account_id'], row['counterparty_id'],
                       weight=row['weight'], count=row['count'])

        # Centrality metrics
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())

        pagerank = nx.pagerank(G, weight='weight', max_iter=100)

        k_bet = min(500, len(G))
        betweenness = nx.betweenness_centrality(G, k=k_bet, normalized=True)

        G_undir = G.to_undirected()
        clustering = nx.clustering(G_undir)

        # Community detection
        try:
            from community import community_louvain
            communities = community_louvain.best_partition(G_undir, weight='weight')
        except Exception:
            # Fallback: assign each node its own community
            communities = {node: i for i, node in enumerate(G_undir.nodes())}

        # Community mule density (train only)
        comm_mule_density: dict[int, float] = {}
        if self.labels_df is not None:
            label_map = dict(zip(self.labels_df['account_id'], self.labels_df['is_mule']))
            comm_members: dict[int, list] = {}
            for node, comm_id in communities.items():
                comm_members.setdefault(comm_id, []).append(node)
            for comm_id, members in comm_members.items():
                mule_flags = [label_map[m] for m in members if m in label_map]
                comm_mule_density[comm_id] = float(np.mean(mule_flags)) if mule_flags else 0.0

        rows = {}
        for acc_id in all_accounts:
            if acc_id not in G:
                rows[acc_id] = zero_row.copy()
                continue

            ind = float(in_deg.get(acc_id, 0))
            outd = float(out_deg.get(acc_id, 0))
            comm_id = communities.get(acc_id, -1)
            density = comm_mule_density.get(comm_id, 0.0)

            rows[acc_id] = {
                'in_degree': ind,
                'out_degree': outd,
                'fan_in_ratio': ind / max(outd, 1),
                'fan_out_ratio': outd / max(ind, 1),
                'betweenness_centrality': float(betweenness.get(acc_id, 0.0)),
                'pagerank': float(pagerank.get(acc_id, 0.0)),
                'community_id': float(comm_id),
                'community_mule_density': density,
                'clustering_coefficient': float(clustering.get(acc_id, 0.0)),
                'total_counterparties': ind + outd,
            }

        records = [rows.get(acc, zero_row) for acc in all_accounts]
        result = pd.DataFrame(records, index=all_accounts)
        result.index.name = 'account_id'

        self.validate_output(result)
        return result

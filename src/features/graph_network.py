import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain

from src.features.base import BaseFeatureGenerator


class GraphNetworkFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, labels_df=None):
        super().__init__('graph_network', 'graph_network')
        self.labels_df = labels_df
        self._community_densities = None

    def get_feature_names(self) -> list[str]:
        return [
            'in_degree', 'out_degree', 'fan_in_ratio', 'fan_out_ratio',
            'betweenness_centrality', 'pagerank', 'community_id',
            'community_mule_density', 'clustering_coefficient', 'total_counterparties',
        ]

    def compute(self, txn, profile=None, cutoff_date=None, **kwargs):
        if cutoff_date is None:
            cutoff_date = pd.Timestamp('2025-06-30')
        cutoff_date = pd.Timestamp(cutoff_date)

        txn = txn.copy()
        txn['transaction_date'] = pd.to_datetime(txn['transaction_date'])
        txn_valid = txn[txn['transaction_date'] <= cutoff_date]

        all_accounts = (
            profile['account_id'].unique() if profile is not None and 'account_id' in profile.columns
            else txn_valid['account_id'].unique()
        )
        result = pd.DataFrame(0.0, index=pd.Index(all_accounts, name='account_id'),
                              columns=self.get_feature_names())

        if 'counterparty_id' not in txn_valid.columns or txn_valid['counterparty_id'].isna().all():
            self.validate_output(result)
            return result

        # Build directed weighted graph with correct edge direction
        # Credits: money flows counterparty → account (inbound)
        # Debits: money flows account → counterparty (outbound)
        txn_edges = txn_valid.dropna(subset=['counterparty_id']).copy()
        txn_edges['src'] = np.where(txn_edges['is_credit'] == 1,
                                     txn_edges['counterparty_id'],
                                     txn_edges['account_id'])
        txn_edges['dst'] = np.where(txn_edges['is_credit'] == 1,
                                     txn_edges['account_id'],
                                     txn_edges['counterparty_id'])

        edges = txn_edges.groupby(['src', 'dst']).agg(
            weight=('transaction_amount', 'sum'),
            count=('transaction_amount', 'count'),
        ).reset_index()

        G = nx.DiGraph()
        for _, row in edges.iterrows():
            G.add_edge(row['src'], row['dst'],
                       weight=row['weight'], count=row['count'])

        if len(G) == 0:
            self.validate_output(result)
            return result

        # Degree metrics
        for node in G.nodes():
            if node in result.index:
                ind = G.in_degree(node)
                outd = G.out_degree(node)
                result.at[node, 'in_degree'] = ind
                result.at[node, 'out_degree'] = outd
                result.at[node, 'fan_in_ratio'] = ind / max(outd, 1)
                result.at[node, 'fan_out_ratio'] = outd / max(ind, 1)
                result.at[node, 'total_counterparties'] = ind + outd

        # PageRank
        try:
            pr = nx.pagerank(G, weight='weight', max_iter=500, tol=1e-4)
        except nx.PowerIterationFailedConvergence:
            # Fallback: use unweighted pagerank which converges more reliably
            pr = nx.pagerank(G, max_iter=500, tol=1e-3)
        for node, val in pr.items():
            if node in result.index:
                result.at[node, 'pagerank'] = val

        # Betweenness centrality (approximate for speed)
        k = min(500, len(G))
        bc = nx.betweenness_centrality(G, k=k)
        for node, val in bc.items():
            if node in result.index:
                result.at[node, 'betweenness_centrality'] = val

        # Community detection on undirected version
        G_undir = G.to_undirected()
        try:
            communities = community_louvain.best_partition(G_undir, weight='weight')
        except Exception:
            communities = {n: 0 for n in G_undir.nodes()}

        for node, comm in communities.items():
            if node in result.index:
                result.at[node, 'community_id'] = comm

        # Clustering coefficient
        clustering = nx.clustering(G_undir)
        for node, val in clustering.items():
            if node in result.index:
                result.at[node, 'clustering_coefficient'] = val

        # Community mule density (train only)
        if self.labels_df is not None and len(self.labels_df) > 0:
            labels = self.labels_df.set_index('account_id')['is_mule'] if 'account_id' in self.labels_df.columns else self.labels_df['is_mule']
            comm_df = pd.DataFrame({'community_id': communities}).rename_axis('account_id')
            comm_df = comm_df.join(labels, how='left')
            density = comm_df.groupby('community_id')['is_mule'].mean()
            self._community_densities = density.to_dict()
            for node in result.index:
                comm = result.at[node, 'community_id']
                result.at[node, 'community_mule_density'] = self._community_densities.get(comm, 0)

        result = result.fillna(0)
        self.validate_output(result)
        return result

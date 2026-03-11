import numpy as np
import pandas as pd
from pathlib import Path

from src.features.velocity import VelocityFeatureGenerator
from src.features.amount_patterns import AmountPatternFeatureGenerator
from src.features.temporal import TemporalFeatureGenerator
from src.features.passthrough import PassThroughFeatureGenerator
from src.features.graph_network import GraphNetworkFeatureGenerator
from src.features.profile_mismatch import ProfileMismatchFeatureGenerator
from src.features.kyc_behavioral import KYCBehavioralFeatureGenerator
from src.features.interactions import InteractionFeatureGenerator


class FeaturePipeline:
    def __init__(self, cutoff_date=None, skip_graph=False):
        self.cutoff = cutoff_date or pd.Timestamp('2025-06-30')
        self.skip_graph = skip_graph

    def run(self, txn, profile, labels=None):
        # Stage 1: Independent groups
        velocity = VelocityFeatureGenerator().compute(txn, profile, self.cutoff)
        amount = AmountPatternFeatureGenerator().compute(txn, profile, self.cutoff)
        temporal = TemporalFeatureGenerator().compute(txn, profile, self.cutoff)
        passthrough = PassThroughFeatureGenerator().compute(txn, profile, self.cutoff)

        # Stage 2: Depends on velocity
        profile_mismatch = ProfileMismatchFeatureGenerator().compute(
            txn, profile, self.cutoff, velocity_features=velocity
        )
        kyc = KYCBehavioralFeatureGenerator().compute(txn, profile, self.cutoff)

        # Stage 3: Graph (expensive, can skip)
        if not self.skip_graph:
            graph = GraphNetworkFeatureGenerator().compute(txn, profile, self.cutoff)
        else:
            graph = pd.DataFrame(
                0.0, index=velocity.index,
                columns=GraphNetworkFeatureGenerator().get_feature_names()
            )
            graph.index.name = 'account_id'

        # Merge all on account_id index
        all_features = velocity.join([amount, temporal, passthrough, profile_mismatch, kyc, graph], how='outer')

        # Stage 4: Interactions (needs complete matrix)
        interactions = InteractionFeatureGenerator().compute_from_features(all_features, profile)
        all_features = all_features.join(interactions, how='left')

        # Clean up
        all_features = all_features.fillna(0).replace([np.inf, -np.inf], 0)
        print(f'Feature matrix: {all_features.shape[0]} accounts x {all_features.shape[1]} features')
        return all_features

    def run_and_save(self, txn, profile, labels, output_path):
        features = self.run(txn, profile, labels)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path)
        return features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-graph', action='store_true')
    parser.add_argument('--output', default='data/processed/features_matrix.parquet')
    args = parser.parse_args()

    from src.data.preprocessor import Preprocessor
    prep = Preprocessor()
    txn, profile, labels = prep.load_and_preprocess()

    pipeline = FeaturePipeline(skip_graph=args.skip_graph)
    pipeline.run_and_save(txn, profile, labels, args.output)

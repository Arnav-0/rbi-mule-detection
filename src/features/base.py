"""Base class for all feature generators."""
from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod


class BaseFeatureGenerator(ABC):
    def __init__(self, name: str, group: str):
        self.name = name
        self.group = group

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return list of feature names this generator produces."""
        ...

    @abstractmethod
    def compute(self, txn: pd.DataFrame, profile: pd.DataFrame, cutoff_date=None, **kwargs) -> pd.DataFrame:
        """Compute features and return DataFrame indexed by account_id."""
        ...

    def validate_output(self, result: pd.DataFrame) -> None:
        """Validate that output has correct columns and index name."""
        assert result.index.name == 'account_id', f"Index name must be 'account_id', got '{result.index.name}'"
        expected = set(self.get_feature_names())
        actual = set(result.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise ValueError(f"[{self.name}] Missing features: {missing}")
        if extra:
            raise ValueError(f"[{self.name}] Unexpected extra features: {extra}")

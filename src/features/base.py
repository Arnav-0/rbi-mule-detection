from abc import ABC, abstractmethod
import pandas as pd
import warnings
from typing import Optional


class BaseFeatureGenerator(ABC):
    def __init__(self, name: str, group: str):
        self.name = name
        self.group = group

    @abstractmethod
    def compute(self, txn: pd.DataFrame, profile: pd.DataFrame,
                cutoff_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        pass

    def get_feature_descriptions(self) -> dict[str, str]:
        return {}

    def validate_output(self, features: pd.DataFrame) -> bool:
        expected = set(self.get_feature_names())
        actual = set(features.columns)
        missing = expected - actual
        if missing:
            raise ValueError(f"{self.name}: Missing features: {missing}")
        if features.isin([float("inf"), float("-inf")]).any().any():
            warnings.warn(f"{self.name}: Replacing infinite values with 0")
            features.replace([float("inf"), float("-inf")], 0, inplace=True)
        return True

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.utils.config import RANDOM_SEED


def split_train_val(profile: pd.DataFrame, test_size=0.2, random_state=RANDOM_SEED):
    train_data = profile[profile["dataset"] == "train"].copy()
    y = train_data["is_mule"]
    train_df, val_df = train_test_split(
        train_data, test_size=test_size, random_state=random_state, stratify=y
    )
    return train_df, val_df


def get_test_accounts(profile: pd.DataFrame) -> pd.DataFrame:
    return profile[profile["dataset"] == "test"].copy()


def create_cv_folds(X, y, n_splits=5, random_state=RANDOM_SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))

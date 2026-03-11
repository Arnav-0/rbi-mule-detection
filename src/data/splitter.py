import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_train_val(profile: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    train_data = profile[profile["dataset"] == "train"].copy()
    y = train_data["is_mule"].fillna(0).astype(int)

    train_df, val_df = train_test_split(
        train_data, test_size=test_size, stratify=y, random_state=random_state
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_test_accounts(profile: pd.DataFrame) -> pd.DataFrame:
    return profile[profile["dataset"] == "test"].reset_index(drop=True)


def create_cv_folds(X, y, n_splits: int = 5, random_state: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))

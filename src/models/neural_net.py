"""PyTorch Neural Network wrapper for mule detection."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModelWrapper

logger = logging.getLogger(__name__)

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class MuleDetectorNN:
    def __init__(self, input_dim: int, hidden_dims: list[int] = None, dropout: float = 0.3):
        import torch.nn as nn

        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state):
        self.net.load_state_dict(state)

    def to(self, device):
        self.net = self.net.to(device)
        return self


class FocalLoss:
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(), targets.float(), reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class NeuralNetWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__(name="neural_net", model_type="neural_network")
        self.scaler = StandardScaler()
        self._nn = None
        self._device = None

    def get_optuna_params(self, trial) -> dict:
        n_layers = trial.suggest_int("n_layers", 2, 5)
        return {
            "n_layers": n_layers,
            "hidden_dims": [
                trial.suggest_int(f"hidden_{i}", 32, 512)
                for i in range(n_layers)
            ],
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }

    def build_model(self, params: dict):
        # Store params; actual model built during fit when input_dim is known
        self._build_params = params

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for NeuralNetWrapper. "
                "Install it with: pip install torch"
            )
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import roc_auc_score

        params = getattr(self, "_build_params", {})
        lr = params.get("lr", 1e-3)
        dropout = params.get("dropout", 0.3)
        batch_size = params.get("batch_size", 128)
        weight_decay = params.get("weight_decay", 1e-4)
        hidden_dims = params.get("hidden_dims", [256, 128, 64, 32])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        X_tr = self.scaler.fit_transform(X_train)
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(np.asarray(y_train), dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

        input_dim = X_tr.shape[1]
        nn_model = MuleDetectorNN(input_dim, hidden_dims=hidden_dims, dropout=dropout)
        nn_model.to(device)

        optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = FocalLoss()

        best_val_auc = -np.inf
        best_state = None
        patience = 15
        no_improve = 0

        for epoch in range(100):
            nn_model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = nn_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:
                nn_model.eval()
                with torch.no_grad():
                    X_v = torch.tensor(self.scaler.transform(X_val), dtype=torch.float32).to(device)
                    logits_v = nn_model(X_v).squeeze().cpu().numpy()
                probs_v = 1 / (1 + np.exp(-logits_v))
                val_auc = roc_auc_score(np.asarray(y_val), probs_v)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.clone() for k, v in nn_model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        if best_state is not None:
            nn_model.load_state_dict(best_state)

        self._nn = nn_model
        self.model = nn_model  # for compatibility
        self.is_fitted = True

    def predict_proba(self, X) -> np.ndarray:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralNetWrapper.")
        import torch
        self._nn.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._nn(X_t).squeeze().cpu().numpy()
        return 1 / (1 + np.exp(-logits))

    def save(self, path: Path):
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._nn.state_dict(), str(path).replace(".joblib", "_weights.pt"))
        joblib.dump({
            "scaler": self.scaler,
            "input_dim": self._nn.input_dim,
            "build_params": getattr(self, "_build_params", {}),
        }, path)

    def load(self, path: Path):
        import torch
        path = Path(path)
        data = joblib.load(path)
        self.scaler = data["scaler"]
        params = data.get("build_params", {})
        hidden_dims = params.get("hidden_dims", [256, 128, 64, 32])
        dropout = params.get("dropout", 0.3)
        nn_model = MuleDetectorNN(data["input_dim"], hidden_dims=hidden_dims, dropout=dropout)
        weights_path = str(path).replace(".joblib", "_weights.pt")
        nn_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self._nn = nn_model
        self._device = torch.device("cpu")
        self.model = nn_model
        self.is_fitted = True

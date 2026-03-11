import numpy as np
import joblib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.models.base import BaseModelWrapper


class MuleDetectorNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class NeuralNetWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__('neural_net', 'neural_network')
        self.scaler = StandardScaler()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nn_model = None
        self.params = None
        self.input_dim = None

    def get_optuna_params(self, trial) -> dict:
        n_layers = trial.suggest_int('n_layers', 2, 5)
        hidden_dims = [
            trial.suggest_int(f'hidden_dim_{i}', 32, 512, log=True)
            for i in range(n_layers)
        ]
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        return {
            'n_layers': n_layers,
            'hidden_dims': hidden_dims,
            'lr': lr,
            'dropout': dropout,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
        }

    def build_model(self, params: dict):
        self.params = params

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.input_dim = X_train_scaled.shape[1]

        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_t = torch.FloatTensor(np.array(y_train).reshape(-1, 1)).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        batch_size = self.params.get('batch_size', 128)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_t = torch.FloatTensor(np.array(y_val).reshape(-1, 1)).to(self.device)
        else:
            X_val_t = None
            y_val_t = None

        # Build model
        self.nn_model = MuleDetectorNN(
            input_dim=self.input_dim,
            hidden_dims=self.params.get('hidden_dims', [256, 128, 64, 32]),
            dropout=self.params.get('dropout', 0.3),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.nn_model.parameters(),
            lr=self.params.get('lr', 1e-3),
            weight_decay=self.params.get('weight_decay', 1e-5),
        )
        criterion = FocalLoss()

        best_val_auc = -1.0
        best_state_dict = None
        patience = 15
        patience_counter = 0

        for epoch in range(100):
            self.nn_model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.nn_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Evaluate on validation set
            if X_val_t is not None:
                self.nn_model.eval()
                with torch.no_grad():
                    val_outputs = self.nn_model(X_val_t)
                    val_probs = torch.sigmoid(val_outputs).cpu().numpy().flatten()
                    val_auc = roc_auc_score(np.array(y_val), val_probs)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state_dict = {k: v.clone() for k, v in self.nn_model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        # Load best model
        if best_state_dict is not None:
            self.nn_model.load_state_dict(best_state_dict)

        self.is_fitted = True

    def predict_proba(self, X) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        self.nn_model.eval()
        with torch.no_grad():
            outputs = self.nn_model(X_t)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        return probs

    def save(self, path: Path):
        path = Path(path)
        torch.save({
            'model_state': self.nn_model.state_dict(),
            'params': self.params,
            'input_dim': self.input_dim,
        }, path)
        joblib.dump(self.scaler, path.with_suffix('.scaler'))

    def load(self, path: Path):
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.params = checkpoint['params']
        self.input_dim = checkpoint['input_dim']
        self.nn_model = MuleDetectorNN(
            input_dim=self.input_dim,
            hidden_dims=self.params.get('hidden_dims', [256, 128, 64, 32]),
            dropout=self.params.get('dropout', 0.3),
        ).to(self.device)
        self.nn_model.load_state_dict(checkpoint['model_state'])
        self.scaler = joblib.load(path.with_suffix('.scaler'))
        self.is_fitted = True

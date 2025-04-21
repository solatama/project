# models/transformer.py

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TransformerDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.fc(x)
        return self.sigmoid(x).squeeze(1)


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 input_dim=None,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1,
                 batch_size=32,
                 lr=1e-3,
                 epochs=20,
                 early_stopping_rounds=5,
                 random_seed=42,
                 device=None):
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.random_seed)

        train_dataset = TransformerDataset(X_train, y_train)
        val_dataset = TransformerDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self.model = SimpleTransformerModel(self.input_dim, self.d_model, self.nhead, self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        best_score = -np.inf
        best_state = None
        patience = 0

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            # Early stopping
            self.model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    outputs = self.model(xb).cpu().numpy()
                    val_preds.extend(outputs)
                    val_targets.extend(yb.numpy())

            val_score = roc_auc_score(val_targets, val_preds)
            if val_score > best_score:
                best_score = val_score
                best_state = self.model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_rounds:
                    break

        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return np.vstack([1 - preds, preds]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def score(self, X, y):
        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)
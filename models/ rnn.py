# models/rnn.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = hn[-1]
        return self.fc(out)

class RNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, seq_len=10, hidden_dim=64, lr=1e-3, epochs=20, batch_size=32, device=None):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _reshape_sequence(self, X):
        n = len(X)
        X_seq = []
        for i in range(self.seq_len, n):
            X_seq.append(X[i - self.seq_len:i])
        return np.array(X_seq)

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        y = y[self.seq_len:]  # Adjust target length
        X = self._reshape_sequence(X)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = RNNModel(input_dim=self.input_dim, hidden_dim=self.hidden_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        self.model.eval()
        X = self.scaler.transform(X)
        X = self._reshape_sequence(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy()
        return np.hstack([(1 - probs), probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
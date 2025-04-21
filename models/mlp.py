# models/mlp.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, hidden_dim=64, lr=1e-3, epochs=20, batch_size=32, device=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self.model = MLPModel(input_dim=self.input_dim, hidden_dim=self.hidden_dim).to(self.device)
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
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy()
        return np.hstack([(1 - probs), probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
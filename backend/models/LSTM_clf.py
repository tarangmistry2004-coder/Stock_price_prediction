import torch.nn as nn
from torch.nn import LSTM, Linear
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error , mean_absolute_error , mean_absolute_percentage_error , accuracy_score , classification_report , r2_score

import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import shap

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = LSTM(
            input_size  = input_size,
            hidden_size = hidden_dim,
            num_layers  = layers,
            batch_first = True,
            dropout     = dropout if layers > 1 else 0.0,  # dropout ignored for single layer
        )
        self.fc = Linear(hidden_dim, 1)
        # self.fc = nn.Sequential(Linear(hidden_dim,1) , nn.Sigmoid())
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # (batch, 1)
 
 
class LSTM_CLFmodel:
 
    def __init__(self):
        self.hidden_dim  = 64
        self.layers      = 3
        self.lookBack    = 60   
        self.epochs      = 200
        self.dropOut     = 0.2
        self.net         = None          
        self.feature_scaler = MinMaxScaler()
        self.df_columns  = None          
        self.target_col  = 'target'
 
    def _prepare_sequence(self, features: np.ndarray, targets: np.ndarray):
        x, y = [], []
        for i in range(self.lookBack, len(features)):
            x.append(features[i - self.lookBack + 1 : i + 1])  
            y.append(targets[i])
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
 
    
    def train(self, df: pd.DataFrame) -> None:
        print('LSTM Classification training ....')
        df = df.dropna().copy()
        self.df_columns = [c for c in df.columns if c != self.target_col]

        features_raw = df[self.df_columns].values        
        targets_raw = df[self.target_col].values         
 
        features_scaled = self.feature_scaler.fit_transform(features_raw)
 
        x, y = self._prepare_sequence(features_scaled, targets_raw)

        ds = TensorDataset(torch.from_numpy(x),torch.from_numpy(y).unsqueeze(1),)   # (N, 1)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
 
        self.net = _LSTMNet(
            input_size = x.shape[2],
            hidden_dim = self.hidden_dim,
            layers  = self.layers,
            dropout = self.dropOut)
        
        
       
        n_down = (y == 0).sum()
        n_up   = (y == 1).sum()
        pos_weight = torch.tensor([n_down / n_up], dtype=torch.float32)

        print(f"  Class dist — Down: {n_down}, Up: {n_up}, pos_weight: {pos_weight.item():.3f}")

        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.005, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5 )

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
 
        self.net.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for xi, yi in loader:
                optimizer.zero_grad()
                pred = self.net(xi)
                loss = loss_fn(pred, yi)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step(total_loss)
 
            if epoch % 10 == 0:
                print(f"epoch: {epoch:3d} | total_loss: {total_loss:.6f} | avg_loss: {total_loss / len(loader):.6f}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.net is  None:
            raise RuntimeError("Call train() before predict()!")
 
        df = df.dropna().copy()
      
        y_true = df[self.target_col].values[self.lookBack:]   # (N-lookBack,)
        

        features_scaled = self.feature_scaler.transform(df[self.df_columns].values)
        targets_raw     = df[self.target_col].values

    
        x, _ = self._prepare_sequence(features_scaled, targets_raw)

        self.net.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            pred_proba = torch.sigmoid(self.net(x_tensor)).detach().numpy().flatten()
            pred_label = (pred_proba >= 0.5).astype(int)

        print('Classification result : ')
        print(f"Accuracy : {accuracy_score(y_true, pred_label) * 100:.1f}%")
        print(f"Baseline (majority class): {max(y_true.mean(), 1-y_true.mean())*100:.1f}%")
        print(classification_report(y_true, pred_label, target_names=['Down','Up']))

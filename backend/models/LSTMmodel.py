import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMmodel:
    def __init__(self):
        self.hidden_dim = 16  
        self.layers = 1             
        self.lookBack = 20
        self.epochs = 50     
        self.dropOut = 0.0

        self.reg_net = None
        self.reg_feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.reg_df_columns = None
        self.target_col = "target"
        
        self.r2 = None
        self.mae = None
        self.sharpe_ratio = None
        self.baseline_mae = None
        self.win_rate = None

    def _prepare_sequence(self, features: np.ndarray, targets: np.ndarray):
        x, y = [], []
        for i in range(self.lookBack, len(features)):
            x.append(features[i - self.lookBack : i])
            y.append(targets[i])
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self, reg_df: pd.DataFrame) -> None:
        print("\n=== LSTM Regression Training Starting ===")
        reg_df = reg_df.dropna().copy()
       
        self.reg_df_columns = [c for c in reg_df.columns if c not in [self.target_col, "Close", "Return_1d", "target_vol"]]

     
        split_idx = int(len(reg_df) * 0.85)
        train_df = reg_df.iloc[:split_idx]
        self.reg_feature_scaler.fit(train_df[self.reg_df_columns].values)

        full_features_scaled = self.reg_feature_scaler.transform(reg_df[self.reg_df_columns].values)
        full_targets = reg_df[self.target_col].values

       
        X_all, y_all = self._prepare_sequence(full_features_scaled, full_targets)

      
        train_sequence_len = split_idx - self.lookBack
        X_train, y_train = X_all[:train_sequence_len], y_all[:train_sequence_len]
        X_val, y_val = X_all[train_sequence_len:], y_all[train_sequence_len:]

       
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False) 
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        self.reg_net = _LSTMNet(
            input_size=X_train.shape[2],
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            dropout=self.dropOut,
        )

        reg_optimizer = torch.optim.AdamW(self.reg_net.parameters(), lr=0.002, weight_decay=1e-4)
        reg_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(reg_optimizer, mode='min', patience=3, factor=0.5)
        reg_loss_fn = nn.HuberLoss(delta=1.0)  
        
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(1, self.epochs + 1):
            self.reg_net.train()
            train_loss = 0.0
            for xi, yi in train_loader:
                reg_optimizer.zero_grad()
                pred = self.reg_net(xi)
                loss = reg_loss_fn(pred, yi)
                loss.backward()
                nn.utils.clip_grad_norm_(self.reg_net.parameters(), max_norm=1.0)
                reg_optimizer.step()
                train_loss += loss.item()

            self.reg_net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_xi, v_yi in val_loader:
                    v_pred = self.reg_net(v_xi)
                    v_loss = reg_loss_fn(v_pred, v_yi)
                    val_loss += v_loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            reg_scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.reg_net.state_dict().copy()

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch: {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if best_model_state is not None:
            self.reg_net.load_state_dict(best_model_state)
        print("Training successfully completed!\n")

    def predict(self, reg_df: pd.DataFrame) -> pd.Series:
        if self.reg_net is None:
            raise RuntimeError("Call train() before predict()!")

        reg_df_clean = reg_df.dropna().copy()
        
        reg_features_scaled = self.reg_feature_scaler.transform(reg_df_clean[self.reg_df_columns].values)
        reg_targets_raw = reg_df_clean[self.target_col].values

        reg_x, reg_y = self._prepare_sequence(reg_features_scaled, reg_targets_raw)

        self.reg_net.eval()
        with torch.no_grad():
            reg_preds = self.reg_net(torch.from_numpy(reg_x)).numpy().flatten()
         
        reg_idx = reg_df_clean.index[self.lookBack :]

        historical_vol = reg_df_clean['target_vol'].iloc[self.lookBack:].values
        reg_preds_returns = reg_preds * historical_vol

        amplitude_dampener = 0.90
        reg_preds_final = reg_preds_returns * amplitude_dampener
        reg_preds_returns = np.clip(reg_preds_final, -0.1, 0.1)

        actual_percentage_returns = reg_df_clean['Return_1d'].iloc[self.lookBack:].values

        y_true_series = pd.Series(actual_percentage_returns, index=reg_idx, name="actual_return")
        y_pred_series = pd.Series(reg_preds_returns, index=reg_idx, name="pred_return")

       
        baseline_mae = mean_absolute_error(actual_percentage_returns, np.zeros_like(actual_percentage_returns))
        self.mae = mean_absolute_error(actual_percentage_returns, reg_preds_returns)
        self.baseline_mae = baseline_mae
        self.r2 = r2_score(actual_percentage_returns, reg_preds_returns)
        
        print("=============== LSTM OOS EVALUATION METRICS ===============")
        print(f"MAE : {self.mae:.6f}  (naive-zero baseline: {self.baseline_mae:.6f})")
        print(f"Return R2 Score : {self.r2:.6f}")
        
        direction_acc = np.mean(np.sign(reg_y) == np.sign(reg_preds))
        self.win_rate = direction_acc * 100
        print(f"Directional Accuracy    : {self.win_rate:.2f}%")
        print("===========================================================")

        strategy_returns = np.sign(reg_preds_returns) * actual_percentage_returns
        trading_days = 252
        daily_rf = 0.065 / trading_days

        excess_returns = strategy_returns - daily_rf
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        self.sharpe_ratio = (
            (mean_excess / std_excess) * np.sqrt(trading_days)
            if std_excess > 0
            else 0.0
        )

        return pd.Series(reg_preds_returns, index=reg_idx, name="lstm_pred_regression")

    def forecast(self, reg_df: pd.DataFrame) -> dict:
        if self.reg_net is None:
            raise RuntimeError("Call train() before forecast()!")

    
        reg_df_clean = reg_df.copy()
        if self.target_col in reg_df_clean.columns:
            reg_df_clean = reg_df_clean.drop(columns=[self.target_col])
        
        reg_df_clean = reg_df_clean.dropna()

        reg_features_scaled = self.reg_feature_scaler.transform(reg_df_clean[self.reg_df_columns].values)
        reg_last_window = reg_features_scaled[-self.lookBack :]

        self.reg_net.eval()
        with torch.no_grad():
            x = torch.from_numpy(reg_last_window[np.newaxis].astype(np.float32))
            raw_scaled_forecast = float(self.reg_net(x).item())  

        current_vol = float(reg_df_clean["target_vol"].iloc[-1])
        unscaled_percentage_return = raw_scaled_forecast * current_vol
        reg_pred_return = np.clip(unscaled_percentage_return, -0.05, 0.05)

        last_date = reg_df_clean.index[-1]
        last_close = float(reg_df_clean["Close"].iloc[-1])
        
        # Safe datetime formatting parser
        last_date_dt = pd.to_datetime(last_date) if isinstance(last_date, str) else last_date
        next_day = (last_date_dt + pd.offsets.BusinessDay(1)).strftime("%Y-%m-%d")
        estimated_close = last_close * (1 + reg_pred_return)
        
        print("\n=============== LSTM FUTURE INFERENCE HORIZON ===============")
        print(f"Last Reference Processing Date : {last_date_dt.date()}")
        print(f"Forecast Target Horizon Execution : {next_day}")
        print(f"Predicted Return Vector           : {reg_pred_return * 100:+.4f}%")
        print(f"Last Observed Close : {last_close:.2f} -> Estimated Target Close: {estimated_close:.2f}")
        print("=============================================================")

        return {
            'model': 'LSTM',
            'r2_score': self.r2,
            'last_date': last_date_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            'last_close': last_close,   
            'prediction': reg_pred_return,
            'est_close': estimated_close,
            'mae': self.mae,
            'baseline_mae': self.baseline_mae,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate
        }
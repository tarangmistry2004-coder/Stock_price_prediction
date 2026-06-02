
# import torch.nn as nn
# from torch.nn import LSTM, Linear
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import root_mean_squared_error , mean_absolute_error , mean_absolute_percentage_error , accuracy_score , classification_report , r2_score

# import matplotlib.pyplot as plt
# import torch
# import pandas as pd
# import numpy as np
 
 
# class _LSTMNet(nn.Module):
#     def __init__(self, input_size: int, hidden_dim: int, layers: int, dropout: float):
#         super().__init__()
#         self.lstm = LSTM(
#             input_size  = input_size,
#             hidden_size = hidden_dim,
#             num_layers  = layers,
#             batch_first = True,
#             dropout     = dropout if layers > 1 else 0.0,  # dropout ignored for single layer
#         )
#         self.fc = Linear(hidden_dim, 1) # for regression 
#         # self.fc = nn.Sequential(Linear(hidden_dim,1) , nn.Sigmoid())
 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])   # (batch, 1)
 
 
# class LSTMmodel:
 
#     def __init__(self):
#         self.hidden_dim  = 32
#         self.layers      = 1
#         self.lookBack    = 20   
#         self.epochs      = 60
#         self.dropOut     = 0.1
     
#         self.reg_net         = None   
#         self.reg_feature_scaler = MinMaxScaler()
       
#         self.reg_df_columns  = None          
#         self.target_col  = 'target'
 
#     def _prepare_sequence(self, features: np.ndarray, targets: np.ndarray):
#         x, y = [], []
#         for i in range(self.lookBack, len(features)):
#             # x.append(features[i - self.lookBack : i])
#             x.append(features[i - self.lookBack  : i  ])  
#             y.append(targets[i])
#         return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
 
    
#     def train(self, reg_df : pd.DataFrame) -> None:
#         print('\n LSTM Regression training')

#         reg_df = reg_df.dropna().copy()
#         self.reg_df_columns = [c for c in reg_df.columns if c in [self.target_col,'Close','Return_1d']]

#         reg_features_raw = reg_df[self.reg_df_columns].values          # (N, n_feat)
#         reg_targets_raw = reg_df[self.target_col].values          # (N,)  unscaled returns
 
#         reg_features_scaled = self.reg_feature_scaler.fit_transform(reg_features_raw)
 
#         reg_x, reg_y = self._prepare_sequence(reg_features_scaled, reg_targets_raw)
    
#         reg_ds = TensorDataset(torch.from_numpy(reg_x),torch.from_numpy(reg_y).unsqueeze(1),)   # (N, 1)
#         reg_ds_loader = DataLoader(reg_ds, batch_size=32, shuffle=False)

#         self.reg_net = _LSTMNet(
#             input_size = reg_x.shape[2],
#             hidden_dim = self.hidden_dim,
#             layers  = self.layers,
#             dropout = self.dropOut,
#         )
               

#         reg_optimizer = torch.optim.Adam(self.reg_net.parameters(), lr=0.0005, weight_decay=1e-4)
#         reg_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(reg_optimizer, mode='min', patience=8, factor=0.5)

#         reg_loss_fn = nn.HuberLoss()
#         # reg_loss_fn = nn.MSELoss()
#         self.reg_net.train()

#         for epoch in range(1, self.epochs + 1):
#             total_loss = 0.0
#             for xi, yi in reg_ds_loader:
#                 reg_optimizer.zero_grad()
#                 pred = self.reg_net(xi)
#                 loss = reg_loss_fn(pred, yi)
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(self.reg_net.parameters(), max_norm=1.0)
#                 reg_optimizer.step()
#                 total_loss += loss.item()
#             reg_scheduler.step(total_loss)
 
#             if epoch % 10 == 0:
#                 print(f"epoch: {epoch:3d} | total_loss: {total_loss:.6f} | avg_loss: {total_loss / len(reg_ds_loader):.6f}")

#         print('training done !!! \n')

#     def predict(self , reg_df : pd.DataFrame) -> pd.Series:
#         if self.reg_net is None:
#             raise RuntimeError("Call train() before predict()!")
 
      
#         reg_df = reg_df.dropna().copy()
#         reg_y_true = reg_df[self.target_col].values[self.lookBack:]

      
#         reg_features_scaled = self.reg_feature_scaler.transform(reg_df[self.reg_df_columns].values)
#         reg_targets_raw     = reg_df[self.target_col].values
 
#         reg_x, reg_y = self._prepare_sequence(reg_features_scaled, reg_targets_raw)
 
#         self.reg_net.eval()

#         with torch.no_grad():
#             reg_preds = self.reg_net(torch.from_numpy(reg_x)).numpy().flatten()   # (N-lookBack,) | for regression
#             reg_preds = np.clip(reg_preds, -0.02, 0.02)
           

     
#         # /// for regression

#         print('\nregression result :')
#         baseline_mae = mean_absolute_error(reg_y_true, np.zeros_like(reg_y_true)) 
#         print("Predict")
#         print(f"  MAE  : {mean_absolute_error(reg_y_true, reg_preds):.6f}  (naive-zero baseline: {baseline_mae:.6f})")
#         print(f"  RMSE : {root_mean_squared_error(reg_y_true, reg_preds):.6f}")
#         close_t = reg_df['Close'].iloc[self.lookBack:-1]
#         future_close_actual = reg_df['Close'].iloc[self.lookBack+1:]

#         print(f'len close_t : {len(close_t)} | len reg_preds : {len(reg_preds)}')
#         close_pred = close_t + (close_t * reg_preds[:-1])
#         compare_result  = pd.DataFrame({'Actual close' : future_close_actual ,
#                                         'pred close' : close_pred,
#                                             'actual return ' : reg_y,
#                                             'pred return': reg_preds})
        
#         print('LSTM Result : \n',compare_result)
#         print(f'R2 score of return: {r2_score(reg_y, reg_preds)}')
#         print(f'R2 score of close price : {r2_score(future_close_actual, close_pred)}')

#         zero_pred = np.zeros_like(reg_y)

#         print(f"Model MAE: {mean_absolute_error(reg_y, reg_preds):.6f}")
#         print(f"Zero MAE : {mean_absolute_error(reg_y, zero_pred):.6f}")

#         direction_acc = np.mean(np.sign(reg_y) == np.sign(reg_preds))
#         print(f"Direction accuracy: {direction_acc * 100:.2f}%")

#         print(f"Return R2: {r2_score(reg_y, reg_preds):.6f}")

#         plt.style.use('dark_background')
#         plt.title('lstm result')
#         plt.subplot(2,2,1)
#         plt.plot(reg_y,label = 'original return',color='green')
#         plt.legend()
#         plt.subplot(2,2,2)
#         plt.plot(reg_preds,label = 'predicted return',color='yellow')
#         plt.legend()
#         plt.subplot(2,2,3)
#         plt.plot(close_t,label = 'original price',color='green' )
#         plt.legend()
#         plt.subplot(2,2,4)
#         plt.plot(close_pred,label = 'predicted price',color='yellow')
#         plt.legend()
#         plt.show()
#   # idx = df.index[self.lookBack:]
#         reg_idx = reg_df.index[self.lookBack:] 

#         return  pd.Series(reg_preds , index=reg_idx , name = ' lstm_pred_regression')
 
#     def forecast(self, reg_df : pd.DataFrame) -> None:
#         if  self.reg_net is None:
#             raise RuntimeError("Call train() before forecast()!")
 
      
#         reg_df_clean = reg_df[self.reg_df_columns].dropna().copy()
 

#         reg_features_scaled = self.reg_feature_scaler.transform(reg_df_clean.values)
#         reg_last_window     = reg_features_scaled[-self.lookBack:]    # (lookBack, n_feat)
 
#         self.reg_net.eval()

#         with torch.no_grad():
#             x = torch.from_numpy(reg_last_window[np.newaxis].astype(np.float32))
#             reg_pred_return = self.reg_net(x).item()    # raw return, same space as target # for regression
#             reg_pred_return = np.clip(reg_pred_return, -0.02, 0.02)


#         # /// for regression

#         last_date = reg_df_clean.index[-1]
#         last_close = float(reg_df_clean[reg_df_clean.index[-1]:]['Close'].iloc[0])
#         next_day = (last_date + pd.offsets.BusinessDay(1)).strftime('%Y-%m-%d')
#         print("Forecast")
#         print(f"Last date : {last_date.date()}")
#         print(f"Forecast for : {next_day}")
#         print(f"Return % : {reg_pred_return * 100:+.4f}%")
#         print(f"Last close : {last_close:.2f}  ->  Est. close: {last_close * (1 + reg_pred_return):.2f}")


import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler , StandardScaler
import torch
import torch.nn as nn
from torch.nn import LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset

# ─── CRITICAL STEP 1: FORCE DETERMINISTIC RANDOM STATE SEED LOCKS ───
# os.environ["PYTHONHASHSEED"] = "42"
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_value:
#     torch.cuda.manual_seed_all(42)

class VariancePreservingLoss(nn.Module):

    def __init__(self, alpha: float = 2.0):
        """Alpha acts as an amplifier factor
        Higher alpha values force the yellow line to stretch out vertically.
        """
        super().__init__()
        self.base_mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Standard Error Penalty (Prevents the model from making random guesses)
        mse_loss = self.base_mse(pred, target)

        # 2. Structural Variance Penalty (Prevents the model from flatlining)
        pred_std = torch.std(pred)
        target_std = torch.std(target)

        # Penalizes the model heavily if it plays it too safe and smooths predictions
        variance_penalty = torch.abs(pred_std - target_std)

        return mse_loss + (self.alpha * variance_penalty)

class _LSTMNet(nn.Module):

    def __init__(self, input_size: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.fc = Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
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
        self.hidden_dim = 12 
        self.layers = 1             
        self.lookBack = 2
        self.epochs = 50     
        self.dropOut = 0.0

        self.reg_net = None
        self.reg_feature_scaler = MinMaxScaler(feature_range=(0,1))
        # self.reg_feature_scaler = StandardScaler()
    
        self.reg_df_columns = None
        self.target_col = "target"

    def _prepare_sequence(self, features: np.ndarray, targets: np.ndarray):
        x, y = [], []
        for i in range(self.lookBack, len(features)):
            x.append(features[i - self.lookBack : i])
            y.append(targets[i])
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self, reg_df: pd.DataFrame) -> None:
        print("\n LSTM Regression training starting...")

        reg_df = reg_df.dropna().copy()
       
        self.reg_df_columns = [c for c in reg_df.columns if c not in [self.target_col, "Close", "Return_1d",'target_vol']]

        reg_features_raw = reg_df[self.reg_df_columns].values
        reg_targets_raw = reg_df[self.target_col].values

        reg_features_scaled = self.reg_feature_scaler.fit_transform(reg_features_raw)
       
        reg_x, reg_y = self._prepare_sequence(reg_features_scaled, reg_targets_raw)

        
        split = int(len(reg_x) * 0.85)
        X_train, X_val = reg_x[:split], reg_x[split:]
        y_train, y_val = reg_y[:split], reg_y[split:]

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_ds , batch_size=32 , shuffle=False)

        self.reg_net = _LSTMNet(
            input_size=reg_x.shape[2],
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            dropout=self.dropOut,
        )

        reg_optimizer = torch.optim.AdamW(self.reg_net.parameters(), lr=0.003, weight_decay=1e-4)
        reg_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(reg_optimizer,mode='min', patience=3, factor=0.5)

        # reg_loss_fn = nn.HuberLoss()  
        # reg_loss_fn = nn.MSELoss()  
        reg_loss_fn = VariancePreservingLoss(alpha=1.5)  
        
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
                nn.utils.clip_grad_norm_(
                    self.reg_net.parameters(), max_norm=1.0
                )
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

            reg_scheduler.step(val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.reg_net.state_dict().copy()

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
                )

        if best_model_state is not None:
            self.reg_net.load_state_dict(best_model_state)
        print("Training successfully completed!\n")

    def predict(self, reg_df: pd.DataFrame) -> pd.Series:
        if self.reg_net is None:
            raise RuntimeError("Call train() before predict()!")

        reg_df = reg_df.dropna().copy()

        reg_features_scaled = self.reg_feature_scaler.transform(
            reg_df[self.reg_df_columns].values
        )
        reg_targets_raw = reg_df[self.target_col].values

        reg_x, reg_y = self._prepare_sequence(
            reg_features_scaled, reg_targets_raw
        )

        self.reg_net.eval()
        with torch.no_grad():
            reg_preds = self.reg_net(torch.from_numpy(reg_x)).numpy().flatten()
         
        reg_idx = reg_df.index[self.lookBack :]


        historical_vol = reg_df['target_vol'].iloc[self.lookBack:].values
        reg_preds_returns = reg_preds * historical_vol

        amplitude_dampener = 0.70
        reg_preds_final = reg_preds_returns * amplitude_dampener

        reg_preds_returns = np.clip(reg_preds_final, -0.5, 0.5)

        actual_percentage_returns = reg_df['Return_1d'].iloc[self.lookBack:].values

        y_true_series = pd.Series(actual_percentage_returns, index=reg_idx, name="actual_return")
        y_pred_series = pd.Series(reg_preds_returns, index=reg_idx, name="pred_return")

        close_t = reg_df["Close"].reindex(reg_idx)
        future_close_actual = close_t * (1 + y_true_series)
        close_pred = close_t * (1 + y_pred_series)

        compare_result = pd.DataFrame(
            {
                "Actual close": future_close_actual,
                "pred close": close_pred,
                "actual return ": y_true_series,
                "pred return": y_pred_series,
            }
        )
        print("LSTM Metric Realization Matrix : \n", compare_result.tail(10))

        baseline_mae = mean_absolute_error(actual_percentage_returns, np.zeros_like(reg_y))
        print("LSTM REGRESSION METRIC")
        print(f"MAE : {mean_absolute_error(actual_percentage_returns, reg_preds_returns):.6f}  (naive-zero baseline: {baseline_mae:.6f})")
        print(f"Return R2 Score : {r2_score(actual_percentage_returns, reg_preds_returns):.6f}")
       
        direction_acc = np.mean(np.sign(reg_y) == np.sign(reg_preds))
        print(f"Directional Accuracy    : {direction_acc * 100:.2f}%")

        plt.style.use("dark_background")
        plt.figure(figsize=(12, 8))
        plt.plot(y_true_series, label="original return", color="green")
        plt.plot(y_pred_series, label="predicted return", color="yellow")
        plt.legend()
        plt.show()

        return pd.Series(reg_preds, index=reg_idx, name="lstm_pred_regression")

    def forecast(self, reg_df: pd.DataFrame) -> None:
        if self.reg_net is None:
            raise RuntimeError("Call train() before forecast()!")

        reg_df_clean = reg_df.dropna().copy()
        reg_features_scaled = self.reg_feature_scaler.transform(
            reg_df_clean[self.reg_df_columns].values
        )

        reg_last_window = reg_features_scaled[-self.lookBack :]

        self.reg_net.eval()
        with torch.no_grad():
        
            x = torch.from_numpy(reg_last_window[np.newaxis].astype(np.float32))

            raw_scaled_forecast = float(self.reg_net(x).item())  

        current_vol = float(reg_df_clean["target_vol"].iloc[-1])
        unscaled_percentage_return = raw_scaled_forecast * current_vol

        reg_pred_return = np.clip(unscaled_percentage_return, -0.015, 0.015)

        last_date = reg_df_clean.index[-1]
        last_close = float(reg_df_clean["Close"].iloc[-1])
        next_day = (last_date + pd.offsets.BusinessDay(1)).strftime("%Y-%m-%d")

        print("LSTM FUTURE INFERENCE HORIZON")
        print(f"Last Reference Processing Date : {last_date.date()}")
        print(f"Forecast Target Horizon Execution : {next_day}")
        print(f"Predicted Return Vector           : {reg_pred_return * 100:+.4f}%")
        print(f"Last Observed Close : {last_close:.2f} -> Estimated Target Close: {last_close * (1 + reg_pred_return):.2f}")

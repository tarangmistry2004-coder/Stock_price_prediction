# from xgboost import XGBRegressor
# import xgboost as xgb

# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from sklearn.metrics import    mean_absolute_error,root_mean_squared_error,mean_absolute_percentage_error , r2_score



# class XGBoostModel:

#     def __init__(self):
#         self.reg_model =  None
#         self.lookBack = 20
#         self.reg_feature_scaler = MinMaxScaler()
#         self.reg_df_columns = None
#         self.target_col = 'target'

#     def _prepare_sequence(self,features: np.ndarray,targets: np.ndarray):

#         x, y = [], []

#         for i in range(self.lookBack, len(features)):
#             seq = features[i - self.lookBack  : i ]
#             x.append(seq.flatten())
#             y.append(targets[i])

#         return (np.array(x, dtype=np.float32),np.array(y, dtype=np.float32))


#     def train(self, reg_df : pd.DataFrame):
#         print('XGB regression traning ...')

#         reg_df = reg_df.dropna().copy()
#         self.reg_df_columns = [c for c in reg_df.columns if c not in ['target','Close','Return_1d']]

#         reg_features_raw = reg_df[self.reg_df_columns].values
#         reg_targets_raw  = reg_df[self.target_col].values


#         reg_feature_scaled = self.reg_feature_scaler.fit_transform(reg_features_raw)

#         reg_x , reg_y = self._prepare_sequence(reg_feature_scaled,reg_targets_raw)

      
       
#         self.reg_model = XGBRegressor(
#             n_estimators = 150,
#             learning_rate = 0.01,
#             max_depth = 3,
#             subsample = 0.6,
#             colsample_bytree = 0.15,
#             reg_alpha = 0.5,
#             reg_lambda = 5.0,
#             objective = 'reg:absoluteerror',
#             random_state = 42,
#         )
#         self.reg_model.fit(reg_x,reg_y)
#         xgb.plot_importance(self.reg_model,max_num_features=100)
#         plt.show()

#         print('training done !!! \n')

#     def predict(self, reg_df : pd.DataFrame) -> pd.Series:

#         if  self.reg_model is  None:
#             raise RuntimeError("Call train() before predict()!")

      
#         reg_df = reg_df.dropna().copy()

#         reg_features_scaled = self.reg_feature_scaler.transform(reg_df[self.reg_df_columns].values)
#         reg_target_raw = reg_df[self.target_col].values

        
#         reg_x , reg_y = self._prepare_sequence(reg_features_scaled,reg_target_raw)

#         reg_preds = self.reg_model.predict(reg_x)

#         reg_idx = reg_df.index[self.lookBack:]
#         y_true_series = pd.Series(reg_y, index=reg_idx, name='actual_return')
#         y_pred_series = pd.Series(reg_preds, index=reg_idx, name='pred_return')

#         close_t = reg_df['Close'].iloc[self.lookBack:]
#         future_close_actual = close_t * (1 + y_true_series)
#         close_pred = close_t * (1 + y_pred_series)

#         print(f'len close_t : {len(close_t)} | len reg_preds : {len(reg_preds)}')
#         compare_result  = pd.DataFrame({'Actual close' : future_close_actual ,
#                                         'pred close' : close_pred,
#                                         'actual return ' : y_true_series,
#                                         'pred return': y_pred_series})
#         # print(future_close_actual , close_pred)
#         compare_result['diff.'] =  compare_result['Actual close']-compare_result['pred close']
#         compare_result['diff. %'] = ((compare_result['pred close'] - compare_result['Actual close'])/compare_result['pred close'])*100 
#         print('XGBOOST Result : \n',compare_result)

#         print(f"reg_preds stats:")
#         print(f"mean : {reg_preds.mean():.6f}")
#         print(f"std : {reg_preds.std():.6f}")
#         print(f"min : {reg_preds.min():.6f}")
#         print(f"max : {reg_preds.max():.6f}")

#         baseline_mae = mean_absolute_error(reg_y, np.zeros_like(reg_y))
#         nonzero      = reg_y != 0
#         print('\nRgression result :')
#         # print('reg y_true : \n',reg_y[-20:] , '\n', 'reg y_pred : \n',reg_preds[-20:])
#         print(f"  MAE  : {mean_absolute_error(reg_y, reg_preds):.6f}  (naive-zero baseline: {baseline_mae:.6f})")
#         print(f'R2 score of return: {r2_score(reg_y, reg_preds)}')
#         print(f'R2 score of close price : {r2_score(future_close_actual, close_pred)}')
#         # print(f"  RMSE : {root_mean_squared_error(reg_y, reg_preds):.6f}")
#         # if nonzero.sum() > 0:
#         #     mape = mean_absolute_percentage_error(reg_y[nonzero], reg_preds[nonzero]) 
#         #     print(f"  MAPE : {mape:.2f}%  (on non-zero targets only)")

#         zero_pred = np.zeros_like(reg_y)

#         print(f"Model MAE: {mean_absolute_error(reg_y, reg_preds):.6f}")
#         print(f"Zero MAE : {mean_absolute_error(reg_y, zero_pred):.6f}")

#         direction_acc = np.mean(np.sign(reg_y) == np.sign(reg_preds))
#         print(f"Direction accuracy: {direction_acc * 100:.2f}%")

#         print(f"Return R2: {r2_score(reg_y, reg_preds):.6f}")

#         plt.style.use('dark_background')  
#         plt.figure(figsize=(12, 8))
#         plt.title('XGBOOST result')
#         plt.subplot(2,2,1)
#         plt.plot(reg_y,label = 'original return',color='green')
#         plt.legend()
#         plt.subplot(2,2,2)
#         plt.plot(reg_preds,label = 'predicted return',color='yellow')
#         plt.legend()
#         plt.subplot(2,2,3)
#         plt.plot(future_close_actual,label = 'original price',color='green' )
#         plt.legend()
#         plt.subplot(2,2,4)
#         plt.plot(close_pred,label = 'predicted price',color='yellow')
#         plt.legend()
#         plt.show()

       

#         return  pd.Series(reg_preds , index=reg_idx , name = 'xgb_pred_regression')

#     def forecast(self,  reg_df : pd.DataFrame):

#         if  self.reg_model is None:
#             raise RuntimeError("Call train() before forecast()!")

#         reg_df_clean = reg_df[self.reg_df_columns].dropna().copy()
#         reg_features_scaled = self.reg_feature_scaler.transform(reg_df_clean.values)
 
#         reg_last_window = reg_features_scaled[-self.lookBack:]       # (lookBack, n_feat)
#         reg_X_forecast  = reg_last_window.flatten().reshape(1, -1)   # (1, lookBack*n_feat)
 
#         reg_pred_return = self.reg_model.predict(reg_X_forecast)[0]      # raw return

#         reg_last_date   = reg_df.index[-1]
#         reg_last_close  = float(reg_df[reg_df_clean.index[-1]:]['Close'].iloc[0])
#         next_day    = (reg_last_date + pd.offsets.BusinessDay(1)).strftime('%Y-%m-%d')
 
#         print("XGBoost Forecast")
#         print(f"Last date : {reg_last_date.date()}")
#         print(f"Forecast from : {next_day}")
#         print(f"Return %  : {reg_pred_return * 100:+.4f}%")
#         print(f"Last close  : {reg_last_close:.2f}  ->  Est. close: {reg_last_close * (1 + reg_pred_return):.2f}")




from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,mean_absolute_percentage_error,r2_score,root_mean_squared_error,
)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import xgboost as xgb


class XGBoostModel:

    def __init__(self):
        self.reg_model = None
        self.lookBack = 20
        self.reg_feature_scaler = MinMaxScaler()
        self.reg_target_scaler = MinMaxScaler(feature_range=(-100,100))
        self.target_col = "target"
        self.engineered_feature_cols = None

    def _prepare_aggregated_features(self, df_features: pd.DataFrame, lookback: int = 20 ):
        df_features = df_features.copy()
        aggregated_features = pd.DataFrame(index=df_features.index)
        immediate_cols = [
            "RSI",
            "MACD_diff",
            "BB_position",
            "ATR",
            "vol_zscore",
            "gap",
            "rel_nifty",
            "sentiment",
            "lag1_return",       
            "lag2_return",     
            "close_to_SMA50",    
            "consec_up",         
            "consec_down"    
        ]
        for col in immediate_cols:
            if col in df_features.columns:
                aggregated_features[f"{col}_t_0"] = df_features[col]
                aggregated_features[f"{col}_t_1"] = df_features[col].shift(1)

      
        rolling_target_cols = [
            "RSI",
            "MACD_diff",
            "vol_zscore",
            "rel_nifty",
            "sentiment",
        ]
        for col in rolling_target_cols:
            if col in df_features.columns:
                aggregated_features[f"{col}_mean_{lookback}d"] = (
                    df_features[col].rolling(window=lookback).mean()
                )
                aggregated_features[f"{col}_std_{lookback}d"] = (
                    df_features[col].rolling(window=lookback).std()
                )

     
        if "Return_1d" in df_features.columns:
            aggregated_features[f"Return_{lookback}d"] = (
                df_features["Return_1d"].rolling(window=lookback).sum()
            )
        if "vol_ratio" in df_features.columns:
            aggregated_features[f"Vol_Sum_{lookback}d"] = (
                df_features["vol_ratio"].rolling(window=lookback).sum()
            )

        
        aggregated_features["target"] = df_features[self.target_col]
        aggregated_features["target_vol"] = df_features["target_vol"]

       
        aggregated_features.dropna(inplace=True)

      
        y_series = aggregated_features["target"]
        X_df = aggregated_features.drop(columns=["target",'target_vol'])
        vol_series = aggregated_features["target_vol"]

        return X_df, y_series , vol_series

    def train(self, reg_df: pd.DataFrame):
        print("XGB regression training starting...")

       
        X_df, y_series , _ = self._prepare_aggregated_features( reg_df, lookback=self.lookBack)

        self.engineered_feature_cols = list(X_df.columns)

       
        reg_x = self.reg_feature_scaler.fit_transform(X_df.values)
        # reg_y = y_series.values
        reg_y = self.reg_target_scaler.fit_transform(y_series.values.reshape(-1,1)).flatten()

        
        self.reg_model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.75,
            colsample_bytree=0.45,
            reg_alpha=0.05,
            reg_lambda=1.5,
            # objective="reg:absoluteerror",
            objective="reg:squarederror",
            # objective="reg:pseudohubererror",
            random_state=42,
        )

        
        self.reg_model.fit(reg_x, reg_y )

        xgb.plot_importance(self.reg_model, max_num_features=25)
        plt.show()
        print("Training successfully completed!\n")

    def predict(self, reg_df: pd.DataFrame) -> pd.Series:
        if self.reg_model is None:
            raise RuntimeError("Call train() before predict()!")

        
        X_df, y_series , vol_series = self._prepare_aggregated_features(
            reg_df, lookback=self.lookBack
        )

        
        X_df = X_df[self.engineered_feature_cols]

        
        reg_x = self.reg_feature_scaler.transform(X_df.values)
        reg_y = y_series.values
        reg_idx = X_df.index

        
        # reg_preds = self.reg_model.predict(reg_x)
        # pred_zscore = self.reg_model.predict(reg_x)

        # reg_preds = pred_zscore * vol_series.values
        # reg_preds = np.clip(reg_preds,-0.0015 , 0.0015)

        # actual_percentage_returns = reg_df['Return_1d'].reindex(reg_idx).values





        pred_zscores = self.reg_model.predict(reg_x)
        pred_zscores = self.reg_target_scaler.inverse_transform(pred_zscores.reshape(-1,1)).flatten()
     
        # raw_percentage_predictions = pred_zscores * vol_series.values
        reg_preds = pred_zscores * vol_series.values
      
        reg_idx = X_df.index
        actual_percentage_returns = reg_df['Return_1d'].reindex(reg_idx).values
        
     
        # actual_std = np.std(actual_percentage_returns)
        # pred_std = np.std(raw_percentage_predictions)
        
       
        # variance_multiplier = (actual_std / pred_std) if pred_std > 0 else 1.0

        # amplitude_dampener = 0.40
        
     
        # reg_preds = reg_preds * 2.0005 
        
       
        reg_preds = np.clip(reg_preds, -0.1, 0.1)
        
        y_true_series = pd.Series(actual_percentage_returns, index=reg_idx, name="actual_return")
        y_pred_series = pd.Series(reg_preds, index=reg_idx, name="pred_return" )

        close_t = reg_df["Close"].reindex(reg_idx)
        future_close_actual = close_t * (1 + y_true_series)
        close_pred = close_t * (1 + y_pred_series)

        print(f"len close_t : {len(close_t)} | len reg_preds : {len(reg_preds)}")
        compare_result = pd.DataFrame(
            {
                "Actual close": future_close_actual,
                "pred close": close_pred,
                "actual return ": y_true_series,
                "pred return": y_pred_series,
            }
        )

        compare_result["diff."] = ( compare_result["Actual close"] - compare_result["pred close"] )
        compare_result["diff. %"] = ( (compare_result["pred close"] - compare_result["Actual close"]) / compare_result["pred close"] ) * 100
        # print("XGBOOST Result : \n", compare_result.tail(10))

        print(f"reg_preds stats:")
        print(f"mean : {reg_preds.mean():.6f}")
        print(f"std : {reg_preds.std():.6f}")
        print(f"min : {reg_preds.min():.6f}")
        print(f"max : {reg_preds.max():.6f}")

        baseline_mae = mean_absolute_error(actual_percentage_returns, np.zeros_like(reg_y))
        print("\nRegression result :")
        print(f"MAE  : {mean_absolute_error(actual_percentage_returns, reg_preds):.6f}  (naive-zero baseline: {baseline_mae:.6f})")
        print(f"R2 score of return: {r2_score(actual_percentage_returns, reg_preds)}")

        direction_acc = np.mean(np.sign(actual_percentage_returns) == np.sign(reg_preds))
        print(f"Direction accuracy: {direction_acc * 100:.2f}%")

        plt.style.use("dark_background")
        plt.figure(figsize=(12, 8))
        plt.plot(actual_percentage_returns, label="original return", color="green",)
        plt.legend()
        plt.plot(reg_preds, label="predicted return", color="yellow",)
        plt.legend()
        plt.show()

        return pd.Series(reg_preds, index=reg_idx, name="xgb_pred_regression")

    # def forecast(self, reg_df: pd.DataFrame):
    #     if self.reg_model is None:
    #         raise RuntimeError("Call train() before forecast()!")

    #     X_df, _ ,vol_series= self._prepare_aggregated_features(
    #         reg_df, lookback=self.lookBack
    #     )

    #     last_row = X_df[self.engineered_feature_cols].tail(1)
    #     reg_X_forecast = self.reg_feature_scaler.transform(last_row.values)

    #     # reg_pred_return = self.reg_model.predict(reg_X_forecast)[0]
    #     pred_zscore = float(self.reg_model.predict(reg_X_forecast)[0])
    #     current_vol = float(vol_series.iloc[-1])
    #     unscaled_return = pred_zscore * current_vol
    #     reg_pred_return = np.clip(unscaled_return, -0.1, 0.1)

    #     last_date = reg_df.index[-1]
    #     last_close = reg_df["Close"].iloc[-1]
    #     estimated_close = last_close * (1 + reg_pred_return)

    #     print(f"\nFORECAST")
    #     print(f"Last Reference Processing Date : {last_date}")
    #     print(f"Predicted Return Execution Vector : {reg_pred_return * 100:.4f}%")
    #     print(f"Last Observed Close : {last_close:.2f} -> Estimated Target Close: {estimated_close:.2f}")

    #     return reg_pred_return

    def forecast(self, reg_df: pd.DataFrame) -> float:
        if self.reg_model is None:
            raise RuntimeError("Call train() before forecast()!")

        X_df, _, vol_series = self._prepare_aggregated_features(
            reg_df, lookback=self.lookBack
        )

        
        X_df = X_df[self.engineered_feature_cols]
        last_row = X_df.tail(1)

        reg_X_forecast = self.reg_feature_scaler.transform(last_row.values)

        raw_scaled_forecast = self.reg_model.predict(reg_X_forecast)

        unscaled_zscore = self.reg_target_scaler.inverse_transform(raw_scaled_forecast.reshape(-1, 1))
        pred_zscore_1d = unscaled_zscore.flatten()

      
        current_vol = float(vol_series.iloc[-1])
        unscaled_percentage_return = float(pred_zscore_1d[0] * current_vol)

      
        reg_pred_return = np.clip(unscaled_percentage_return, -0.1, 0.1)

        last_date = reg_df.index[-1]
        last_close = float(reg_df["Close"].iloc[-1])
        estimated_close = last_close * (1 + reg_pred_return)

        print(f"\nFORECAST")
        print(f"Last Reference Processing Date : {last_date}")
        print(f"Predicted Return Execution Vector : {reg_pred_return * 100:+.4f}%")
        print(f"Last Observed Close : {last_close:.2f} -> Estimated Target Close: {estimated_close:.2f}")

        return reg_pred_return


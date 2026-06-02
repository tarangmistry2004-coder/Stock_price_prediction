import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

class LightGBMModel:

    def __init__(self):
        self.reg_model = None
        self.lookBack = 20
        self.reg_feature_scaler = MinMaxScaler()
        self.reg_target_scaler = MinMaxScaler(feature_range=(-100,100)) 
        self.target_col = "target"
        self.engineered_feature_cols = None

    def _prepare_aggregated_features(self, df_features: pd.DataFrame, lookback: int = 20):
        df_features = df_features.copy()
        aggregated_features = pd.DataFrame(index=df_features.index)

        immediate_cols = [
            "RSI",
            "MACD_diff",
            "ATR",
            "vol_zscore",
            "gap",
            "rel_nifty",
            "closing_marubozu",
            "vol_stretch",
            "lag1_return",
            "lag2_return",
            'ADX' ,
            'ADX_slope',
            'Stochastic_K',
            'Stochastic_D'  
        ]
        for col in immediate_cols:
            if col in df_features.columns:
                aggregated_features[f"{col}_t_0"] = df_features[col]
                aggregated_features[f"{col}_t_1"] = df_features[col].shift(1)

        rolling_target_cols = ["RSI", "MACD_diff", "vol_zscore", "rel_nifty"]
        for col in rolling_target_cols:
            if col in df_features.columns:
                aggregated_features[f"{col}_mean_{lookback}d"] = (
                    df_features[col].rolling(window=lookback).mean()
                )
                aggregated_features[f"{col}_std_{lookback}d"] = (
                    df_features[col].rolling(window=lookback).std()
                )

        aggregated_features["target"] = df_features[self.target_col]
        aggregated_features["target_vol"] = df_features["target_vol"]
        aggregated_features["nifty_ret"] = df_features["nifty_ret"] if "nifty_ret" in df_features.columns else 0.0
        aggregated_features.dropna(inplace=True)

        y_series = aggregated_features["target"]
        vol_series = aggregated_features["target_vol"]
        bench_series = aggregated_features["nifty_ret"]
        X_df = aggregated_features.drop(columns=["target", "target_vol"])

        return X_df, y_series, vol_series , bench_series

    def compute_institutional_metrics(self,actual_ret: np.ndarray, pred_ret: np.ndarray, benchmark_ret: np.ndarray) -> dict:
      
       
        trading_days = 252

        
        daily_rf = 0.065 / trading_days

        #  SHARPE RATIO CALCULATIONS ───
       
        strategy_returns = np.sign(pred_ret) * actual_ret

        excess_returns = strategy_returns - daily_rf
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        sharpe = (
            (mean_excess / std_excess) * np.sqrt(trading_days)
            if std_excess > 0
            else 0.0
        )

        #  INFORMATION RATIO (IR) CALCULATIONS ───
        # Active Return (Alpha) = Strategy Return - Benchmark Index Return
        active_returns = strategy_returns - benchmark_ret

        mean_active = np.mean(active_returns)
        tracking_error = np.std(active_returns)  # Volatility of active returns

        information_ratio = (
            (mean_active / tracking_error) * np.sqrt(trading_days)
            if tracking_error > 0
            else 0.0
        )

        #  OUT-OF-SAMPLE STABILITY METRICS ───
        win_rate = np.mean(np.sign(actual_ret) == np.sign(pred_ret)) * 100

        # Profit Factor calculation
        gains = strategy_returns[strategy_returns > 0].sum()
        losses = np.abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = (gains / losses) if losses > 0 else float("inf")

        return {
            "Annualized Sharpe Ratio": sharpe,
            "Annualized Information Ratio": information_ratio,
            "Out-of-Sample Win Rate": win_rate,
            "System Profit Factor": profit_factor,
        }


    def train(self, train_df: pd.DataFrame):
        print("LightGBM regression training starting...")
        
        X_df, y_series, _ , _= self._prepare_aggregated_features(train_df, lookback=self.lookBack)
        self.engineered_feature_cols = list(X_df.columns)

        reg_x = self.reg_feature_scaler.fit_transform(X_df.values)
        reg_y = self.reg_target_scaler.fit_transform(y_series.values.reshape(-1, 1)).flatten()

        self.reg_model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=1.5,
            objective="regression",
            random_state=42,
            verbose = -1
        )
        self.reg_model.fit(reg_x, reg_y)
        print("LightGBM training completed successfully!\n")

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if self.reg_model is None:
            raise RuntimeError("Call train() before predict()!")

        X_df, y_series, vol_series , bench_series = self._prepare_aggregated_features(test_df, lookback=self.lookBack)
        X_df = X_df[self.engineered_feature_cols]

        reg_x = self.reg_feature_scaler.transform(X_df.values)
        reg_idx = X_df.index

        raw_preds = self.reg_model.predict(reg_x)
        pred_zscores_1d = self.reg_target_scaler.inverse_transform(raw_preds.reshape(-1, 1)).flatten()

        reg_preds = pred_zscores_1d * vol_series.values
        reg_preds = np.clip(reg_preds, -0.05, 0.05)

        actual_percentage_returns = (test_df["Return_1d"].reindex(reg_idx).values)

        y_true_series = pd.Series(actual_percentage_returns, index=reg_idx, name="actual_return")
        y_pred_series = pd.Series(reg_preds, index=reg_idx, name="pred_return")

        close_t = test_df["Close"].reindex(reg_idx)
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

        compare_result["diff."] = (compare_result["Actual close"] - compare_result["pred close"])
        compare_result["diff. %"] = ((compare_result["pred close"] - compare_result["Actual close"])/ compare_result["pred close"]) * 100
        # print("LightGBM Result Verification : \n", compare_result.tail(10))

        baseline_mae = mean_absolute_error(actual_percentage_returns, np.zeros_like(actual_percentage_returns))
        print("\nLightGBM Analysis Report :")
        print(f"MAE  : {mean_absolute_error(actual_percentage_returns, reg_preds):.6f}  (baseline: {baseline_mae:.6f})")
        # print(f"R2 score of return: {r2_score(actual_percentage_returns, reg_preds)}")

        # direction_acc = np.mean(np.sign(actual_percentage_returns) == np.sign(reg_preds))
        # print(f"Direction accuracy: {direction_acc * 100:.2f}%")

        metrics = self.compute_institutional_metrics(
            actual_ret=actual_percentage_returns,
            pred_ret=reg_preds,
            benchmark_ret=bench_series.values
        )

        print(f"\n" + "="*15 + f" {type(self).__name__} OOS METRICS " + "="*15)
        print(f"  Return R2 Score               : {r2_score(actual_percentage_returns, reg_preds):.6f}")
        print(f"  Annualized Sharpe Ratio       : {metrics['Annualized Sharpe Ratio']:.4f}")
        print(f"  Annualized Information Ratio  : {metrics['Annualized Information Ratio']:.4f}")
        print(f"  Out-of-Sample Win Rate        : {metrics['Out-of-Sample Win Rate']:.2f}%")
        print(f"  System Profit Factor          : {metrics['System Profit Factor']:.4f}")
        print("=" * 55 + "\n")


        plt.style.use("dark_background")
        plt.figure(figsize=(12, 6))
        plt.plot(actual_percentage_returns,label="original return",color="green")
        plt.plot(reg_preds, label="predicted return", color="yellow", )
        plt.title("LightGBM Out-of-Sample Return Waves")
        plt.legend()
        plt.show()

        return pd.Series(reg_preds, index=reg_idx, name="lgb_pred_regression")

    def forecast(self, test_df: pd.DataFrame) -> float:
        if self.reg_model is None:
            raise RuntimeError("Call train() before forecast()!")

        X_df, _, vol_series , _ = self._prepare_aggregated_features(test_df, lookback=self.lookBack)
        last_row = X_df[self.engineered_feature_cols].tail(1)
        reg_X_forecast = self.reg_feature_scaler.transform(last_row.values)

        raw_scaled_forecast = self.reg_model.predict(reg_X_forecast)
        pred_zscore_1d = self.reg_target_scaler.inverse_transform(raw_scaled_forecast.reshape(-1, 1)).flatten()

        current_vol = float(vol_series.iloc[-1])
        unscaled_return = float(pred_zscore_1d * current_vol)
        reg_pred_return = np.clip(unscaled_return, -0.015, 0.015)

        last_date = test_df.index[-1]
        last_close = test_df["Close"].iloc[-1]
        estimated_close = last_close * (1 + reg_pred_return)

        print(f"\nLIGHTGBM FORECAST")
        print(f"Last Reference Processing Date : {last_date}")
        print(f"Predicted Return Execution Vector : {reg_pred_return * 100:.4f}%")
        print(f"Last Observed Close : {last_close:.2f} -> Estimated Target Close: {estimated_close:.2f}")

        return reg_pred_return

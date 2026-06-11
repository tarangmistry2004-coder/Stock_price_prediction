import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import QuantileLoss, TemporalFusionTransformer, TimeSeriesDataSet

warnings.filterwarnings("ignore")

class TFTQuantModel:

    def __init__(self):
        self.reg_model = None
        self.lookBack = 20  
        self.reg_feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.reg_target_scaler = MinMaxScaler(feature_range=(-100, 100))
        self.target_col = "target"
        self.engineered_feature_cols = None
        self.training_dataset = None  

    def _prepare_tft_dataframe(self, df_features: pd.DataFrame, is_training: bool = True):
        df = df_features.copy()
        df = df.dropna().reset_index()

        df["time_idx"] = df.index
        df["group_id"] = "STOCK_ASSET"

        if is_training:
            self.engineered_feature_cols = [c for c in df.columns if c not in [
                    "Date",
                    "time_idx",
                    "group_id",
                    self.target_col,
                    "target_vol",
                    "Close",
                    "Return_1d",
                ]
            ]

        if is_training:
            df[self.engineered_feature_cols] = (self.reg_feature_scaler.fit_transform(df[self.engineered_feature_cols].values))
            df[self.target_col] = (self.reg_target_scaler.fit_transform(df[self.target_col].values.reshape(-1, 1)).flatten())
        else:
            df[self.engineered_feature_cols] = (self.reg_feature_scaler.transform(df[self.engineered_feature_cols].values))
            

        return df

    def train(self, train_df: pd.DataFrame):
        print("Temporal Fusion Transformer (TFT) training starting...")

       
        df_processed = self._prepare_tft_dataframe(train_df, is_training=True)

       
        self.training_dataset = TimeSeriesDataSet(
            df_processed,
            time_idx="time_idx",
            target=self.target_col,
            group_ids=["group_id"],
            min_encoder_length=self.lookBack,
            max_encoder_length=self.lookBack,
            min_prediction_length=1,
            max_prediction_length=1, 
            time_varying_known_reals=["time_idx"],  
            time_varying_unknown_reals=[self.target_col]+ self.engineered_feature_cols,
            target_normalizer=None,  
        )

        # Build PyTorch DataLoaders
        train_dataloader = self.training_dataset.to_dataloader(batch_size=32, shuffle=False)

        self.reg_model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=0.001,
            hidden_size=16, 
            attention_head_size=2,  
            dropout=0.3,
            loss=QuantileLoss(), 
            reduce_on_plateau_patience=4,
        )

    
        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="cpu", 
            enable_model_summary=True,
            callbacks=[
                EarlyStopping(monitor="train_loss", patience=5),
                # LearningRateMonitor(),
            ],
            logger=False,
        )

       
        trainer.fit(self.reg_model, train_dataloaders=train_dataloader)
        print("TFT multi-head network training completely completed!\n")

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if self.reg_model is None:
            raise RuntimeError("Call train() before predict()!")

        df_processed = self._prepare_tft_dataframe(test_df, is_training=False)

        # Re-build out-of-sample prediction metadata datasets
        validation_dataset = TimeSeriesDataSet.from_dataset( self.training_dataset, df_processed)
        val_dataloader = validation_dataset.to_dataloader(batch_size=32, shuffle=False)


        raw_predictions = self.reg_model.predict(val_dataloader, mode="prediction", return_x=False)
        scaled_preds_1d = raw_predictions.numpy().flatten()

        pred_zscores_1d = self.reg_target_scaler.inverse_transform(scaled_preds_1d.reshape(-1, 1)).flatten()

      
        aligned_df = df_processed.iloc[self.lookBack :].copy()
        reg_idx = pd.to_datetime(aligned_df["Date"].values)

        reg_preds = pred_zscores_1d * aligned_df["target_vol"].values
        reg_preds = np.clip(reg_preds, -0.05, 0.05)

        actual_percentage_returns = aligned_df["Return_1d"].values
        baseline_mae = mean_absolute_error(actual_percentage_returns, np.zeros_like(actual_percentage_returns))

        print("\n" + "=" * 15 + " TFT TRANSFORMER METRIC BRIEF " + "=" * 15)
        print(f"Return R2 Score : {r2_score(actual_percentage_returns, reg_preds):.6f}")
        print(f"Model MAE : {mean_absolute_error(actual_percentage_returns, reg_preds):.6f} (Baseline: {baseline_mae:.6f})")
        print(f"  Directional Accuracy : {np.mean(np.sign(actual_percentage_returns) == np.sign(reg_preds)) * 100:.2f}%")

        plt.style.use("dark_background")
        plt.figure(figsize=(12, 6))
        plt.plot(actual_percentage_returns, label="original return",color="green",)
        plt.plot(reg_preds, label="predicted return", color="yellow")
        plt.title("TFT Multi-Head Self-Attention Return Waves")
        plt.legend()
        plt.show()

        return pd.Series(reg_preds, index=reg_idx, name="tft_pred_regression")

    def forecast(self, test_df: pd.DataFrame) -> float:
        if self.reg_model is None:
            raise RuntimeError("Call train() before forecast()!")

       
        trailing_buffer_df = test_df.tail(self.lookBack + 5)
        df_processed = self._prepare_tft_dataframe(trailing_buffer_df, is_training=False)

        validation_dataset = TimeSeriesDataSet.from_dataset(self.training_dataset, df_processed)
        val_dataloader = validation_dataset.to_dataloader( batch_size=1, shuffle=False)

        raw_scaled_forecast = self.reg_model.predict(val_dataloader, mode="prediction")
        pred_zscore_1d = self.reg_target_scaler.inverse_transform(raw_scaled_forecast.numpy().reshape(-1, 1)).flatten()

        current_vol = float(test_df["target_vol"].iloc[-1])
        unscaled_return = float(pred_zscore_1d[-1] * current_vol)
        reg_pred_return = np.clip(unscaled_return, -0.015, 0.015)

        last_date = test_df.index[-1]
        last_close = test_df["Close"].iloc[-1]
        estimated_close = last_close * (1 + reg_pred_return)

        print(f"\nTEMPORAL FUSION TRANSFORMER (TFT) FORECAST")
        print(f"Last Reference Processing Date : {last_date}")
        print( f"Predicted Return Execution Vector : {reg_pred_return * 100:.4f}%")
        print(f"Last Observed Close : {last_close:.2f} -> Estimated Target Close: {estimated_close:.2f}")

        return reg_pred_return

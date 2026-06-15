import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping , ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

def prepare_tft_data(nifty_df, encoder_length=21, prediction_length=1, batch_size=64):
    processed_df = nifty_df.copy()
    
    processed_df['Ticker'] = processed_df['Ticker'].astype(str)
    processed_df['Sector'] = processed_df['Sector'].astype(str)
    processed_df['DayOfWeek'] = processed_df['DayOfWeek'].astype(str)
    processed_df['Month'] = processed_df['Month'].astype(str)
    
    real_features = ['Return_1d', 'Return_5d', 'Z_Score_20', 'Volume_Ratio', 'RSI_14', 'Target']
    for col in real_features:
        processed_df[col] = processed_df[col].astype('float32')
        
    processed_df = processed_df.sort_values(by=['Ticker', 'TimeIndex']).reset_index(drop=True)

    tft_blueprint = TimeSeriesDataSet(
        processed_df,
        time_idx="TimeIndex",         
        target="Target",              
        group_ids=["Ticker"],        
        
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,     
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length, 
        
        static_categoricals=["Ticker", "Sector"],
        static_reals=[],
        time_varying_known_categoricals=["DayOfWeek", "Month"],
        time_varying_known_reals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "Return_1d",
            "Return_5d",
            "Z_Score_20",
            "Volume_Ratio",
            "RSI_14"
        ],
        
        target_normalizer=GroupNormalizer(groups=["Ticker"]),
        allow_missing_timesteps=False  
    )

    test_window = 60  
    val_window = 60  
    
    max_idx = processed_df["TimeIndex"].max()
    test_cutoff = max_idx - test_window
    val_cutoff = test_cutoff - val_window
    
   
    train_df = processed_df[processed_df["TimeIndex"] <= val_cutoff]
    
    val_df = processed_df[
        (processed_df["TimeIndex"] > val_cutoff - encoder_length) & 
        (processed_df["TimeIndex"] <= test_cutoff)
    ]
    
    test_df = processed_df[processed_df["TimeIndex"] > test_cutoff - encoder_length]
    train_dataset = TimeSeriesDataSet.from_dataset(tft_blueprint, train_df, predict=False)
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_df, predict=True, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test_df, predict=True, stop_randomization=True)

   
    train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    test_loader = test_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    
    return train_loader, val_loader, test_loader, train_dataset

def initialize_tft_and_trainer(train_dataset):
   
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=1e-3,             
        hidden_size=32,                 
        attention_head_size=4,          # Number of attention heads (interpretable links)
        dropout=0.2,                    
        hidden_continuous_size=16,     
        loss=QuantileLoss([0.1, 0.5, 0.9]), 
        reduce_on_plateau_patience=4    
    )
    
    
    print(f"Trainable Parameters: {sum(p.numel() for p in tft.parameters() if p.requires_grad):,}")

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=10, 
        verbose=True, 
        mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best-tft-nifty50-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min"
    )

   
    logger = TensorBoardLogger("tft_logs", name="nifty50_forecast")

   
    trainer = pl.Trainer(
        max_epochs=30,               
        accelerator="auto",             # Automatically picks GPU if available, else CPU
        devices=1,                      # Number of computing units to deploy
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        gradient_clip_val=0.1          # Clips exploding gradients 
    )
    
    return tft, trainer


def run_predictions_and_plot(best_checkpoint_path, test_dataloader, df, asset_ticker="RELIANCE.NS"):
    index_ledger = test_dataloader.dataset.decoded_index
    ticker_indices = index_ledger[index_ledger["Ticker"] == asset_ticker]

    if ticker_indices.empty:
        print(f"Warning: Ticker '{asset_ticker}' was not found in the test dataset split!")
        print(f"Available tickers are: {index_ledger['Ticker'].unique().tolist()}")
        return
        
    ticker_idx = ticker_indices.index[0]
    mapping_info = index_ledger.iloc[ticker_idx]
    print(f"Ticker Name found in set on index {ticker_idx}:{mapping_info['Ticker']}")


    time_idx_first = mapping_info["time_idx_first"]
    time_idx_pred = mapping_info["time_idx_first_prediction"]

  
    ticker_df = df[df["Ticker"] == asset_ticker].sort_values("TimeIndex")
    date_col = "Date" if "Date" in df.columns else "date"
    
    last_history_row = ticker_df[ticker_df["TimeIndex"] == (time_idx_pred - 1)]
    pred_row = ticker_df[ticker_df["TimeIndex"] == time_idx_pred]
    
    try:
        last_date_raw = last_history_row[date_col].values[0]
        pred_date_raw = pred_row[date_col].values[0]
        
        last_date_str = str(last_date_raw).split("T")[0] 
        pred_date_str = str(pred_date_raw).split("T")[0]
    except IndexError:
        last_date_str = f"Day Index {time_idx_pred - 1}"
        pred_date_str = f"Day Index {time_idx_pred}"

    model = TemporalFusionTransformer.load_from_checkpoint(best_checkpoint_path)
    model.eval() 
    
    raw_predictions = model.predict(test_dataloader, mode="raw", return_x=True , return_y=True)


    preds_tensor = raw_predictions.output.prediction
    if len(preds_tensor.shape) == 3:
    #  (samples, time_steps, quantiles) 
        preds = preds_tensor[:, 0, 1].cpu().numpy().flatten()
    else:
    # (samples, quantiles) 
        preds = preds_tensor[:, 1].cpu().numpy().flatten()
    
    if isinstance(raw_predictions.y, tuple):
        actuals = raw_predictions.y[0].cpu().numpy().flatten()
    else:
        actuals = raw_predictions.y.cpu().numpy().flatten()

    
    tft_mae = mean_absolute_error(actuals, preds)
    tft_r2 = r2_score(actuals, preds)

   
    correct_direction = np.sign(actuals) == np.sign(preds)
    directional_accuracy = np.mean(correct_direction) * 100

 
    print("=============== TFT OOS EVALUATION METRICS ===============")
    print(f"MAE : {tft_mae:.6f}")
    print(f"Return R2 Score : {tft_r2:.6f}")
    print(f"Directional Accuracy    : {directional_accuracy:.2f}%")
    print("==========================================================")
   
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model.plot_prediction(
        raw_predictions.x, 
        raw_predictions.output, 
        idx=ticker_idx, 
        ax=ax,
        plot_attention=True, 
        add_loss_to_title=False
    )
    
    ax.set_ylabel("Normalized Returns (Actual & Predicted)", color="blue")
    
    if len(fig.axes) > 1:
        fig.axes[1].set_ylabel("Model Attention Weights", color="gray")
    
    plt.title(f"TFT Return Forecast: {asset_ticker}\nHistory Window End: {last_date_str} | Target Prediction Date: {pred_date_str}", 
              fontsize=13, fontweight='bold')
    plt.xlabel("Sequential Forecasting Steps")
    # plt.show()
    
   
    history_df = ticker_df[(ticker_df["TimeIndex"] >= time_idx_first) & (ticker_df["TimeIndex"] < time_idx_pred)]
    if history_df.empty:
        print(f"Error: Could not find historical rows in dataframe for time index range {time_idx_first} to {time_idx_pred}")
        return
    
   
    last_known_price = history_df["Close"].values[-1]
    
   
    actual_price = pred_row["Close"].values[0] if not pred_row.empty else None

   
    pred_tensor = raw_predictions.output.prediction[ticker_idx, 0] 
    num_quantiles = pred_tensor.shape[-1]
    
    # Extract lower bounds (10th percentile), median (50th percentile), and upper bounds (90th percentile)
    p10_return = pred_tensor[0].item()                         
    p50_return = pred_tensor[num_quantiles // 2].item()         
    p90_return = pred_tensor[-1].item()                        
    
    # Reverse transformation: Price = Last Price * (1 + Return)
    p10_price = last_known_price * (1 + p10_return)
    p50_price = last_known_price * (1 + p50_return)
    p90_price = last_known_price * (1 + p90_return)
    
    print(f"Real Price Verification Summary: {asset_ticker}")
    print(f" Last History Close ({last_date_str}):₹{last_known_price:.2f}")
    if actual_price is not None:
        print(f"Actual Close Tomorrow ({pred_date_str}):   ₹{actual_price:.2f}")
    print(f" TFT Predicted Close ({pred_date_str}):     ₹{p50_price:.2f}")
    print(f"Volatility Risk Range (10%-90%):  ₹{p10_price:.2f} to ₹{p90_price:.2f}")

def run_live_future_prediction(best_checkpoint_path, df, asset_ticker="RELIANCE.NS"):
   
    date_col = "Date" if "Date" in df.columns else "date"
    processed_df = df.copy()
    processed_df[date_col] = pd.to_datetime(processed_df[date_col])
    
    
    ticker_df = processed_df[processed_df["Ticker"] == asset_ticker].sort_values("TimeIndex")
    if ticker_df.empty:
        print(f"Error: Ticker '{asset_ticker}' not found in dataframe.")
        return
        
    last_row = ticker_df.iloc[-1]
    last_date = last_row[date_col]
    last_idx = last_row["TimeIndex"]
    last_known_price = last_row["Close"]
    
    #
    next_date = last_date + pd.tseries.offsets.BDay(1)
    next_idx = last_idx + 1
    
    # Format dates as clean strings for visual printing
    last_date_str = last_date.strftime('%Y-%m-%d')
    next_date_str = next_date.strftime('%Y-%m-%d')
    
    print(f"Latest Market Baseline Date : {last_date_str} )")
    print(f"Generating Live Forecast For: {next_date_str} )")
   
    
   
    future_row = last_row.copy()
    future_row[date_col] = next_date
    future_row["TimeIndex"] = next_idx
    future_row["DayOfWeek"] = str(next_date.dayofweek)
    future_row["Month"] = str(next_date.month)
    future_row["Target"] = 0.0  # Placeholder target value for the model to overwrite
    
 
    extended_df = pd.concat([processed_df, pd.DataFrame([future_row])], ignore_index=True)
    
    #
    extended_df['Ticker'] = extended_df['Ticker'].astype(str)
    extended_df['Sector'] = extended_df['Sector'].astype(str)
    extended_df['DayOfWeek'] = extended_df['DayOfWeek'].astype(str)
    extended_df['Month'] = extended_df['Month'].astype(str)
    
    model = TemporalFusionTransformer.load_from_checkpoint(best_checkpoint_path)
    model.eval()
    
    future_dataset = TimeSeriesDataSet.from_parameters(
        model.dataset_parameters, 
        extended_df, 
        predict=True, 
        stop_randomization=True
    )
    future_loader = future_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    
    
    raw_predictions = model.predict(future_loader, mode="raw", return_x=True, return_y=False)
    
   
    index_ledger = future_loader.dataset.decoded_index
    ticker_indices = index_ledger[index_ledger["Ticker"] == asset_ticker]
    ticker_idx = ticker_indices.index[0]
    
   
    pred_tensor = raw_predictions.output.prediction[ticker_idx, 0]
    num_quantiles = pred_tensor.shape[-1]
    
    p10_return = pred_tensor[0].item()                         # 10th percentile
    p50_return = pred_tensor[num_quantiles // 2].item()         # 50th percentile (Median)
    p90_return = pred_tensor[-1].item()                        # 90th percentile
    
  
    p10_price = last_known_price * (1 + p10_return)
    p50_price = last_known_price * (1 + p50_return)
    p90_price = last_known_price * (1 + p90_return)
    
  
    print(f"LIVE TARGET PREDICTIONS FOR: {asset_ticker}")
    print(f"Last History Close Price ({last_date_str}) : ₹{last_known_price:.2f}")
    print(f"Predicted Return Vector : {p50_return * 100:+.4f}%")
    print(f"TFT Estimated Target Close ({next_date_str}): ₹{p50_price:.2f}")
    print(f"Volatility Risk Range (10%-90%) : ₹{p10_price:.2f} to ₹{p90_price:.2f}")
   
    
    return {
        'model': 'TFT',
        'ticker' : asset_ticker,
        'last_date' : last_date ,
        'last_close' : last_known_price,   
        'prediction':p50_return,
        'est_close' : p50_price,
        'risk_range' : {
            'low': p10_price,
            'high' : p90_price,
            }
        }
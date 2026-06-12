from DataCollection.DataFetch import get_stock_data
from DataCollection.NIFTYDataFetch import get_nifty_data

from models.LSTMmodel import LSTMmodel
from models.XGBoostmodel import XGBoostModel
from models.Prophetmodel import ProphetModel
from models.LSTM_clf import LSTM_CLFmodel
from models.XGBoost_clf import XGBoost_CLFModel
from models.TFT import TFTQuantModel
from models.catmodel import CatBoostModel
from models.LightGBMmodel import LightGBMModel

from BT.test1 import ModelBacktester

from models.NIftyTFT import prepare_tft_data , initialize_tft_and_trainer ,run_predictions_and_plot , run_live_future_prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


model = LSTMmodel()
model2 = XGBoostModel()
model3 = ProphetModel()
lstm_clf_model = LSTM_CLFmodel()
xgb_clf_model = XGBoost_CLFModel()
cat_boost = CatBoostModel()
light_gbm = LightGBMModel()

# classification_ds , data_for_backtesing = get_stock_data('reliance.ns' , 'max' , True)
regression_ds , data_for_backtesting = get_stock_data('reliance.ns','max')
# regression_ds , data_for_backtesing = get_stock_data('TCS.bo','max')

# print(regression_ds.isna().sum())

# regression_ds.to_csv('RS_reg.csv')
# classification_ds.to_csv('RS_cls.csv')

# corr = regression_ds.corr(numeric_only=True).round(2)
# plt.figure(figsize=(20,20))
# # sn.heatmap(corr,annot=True,cmap='coolwarm')

# corr = (regression_ds.select_dtypes(include=["number"])
#       .corr()["target"]
#       .drop("target")
#       .sort_values(key=abs, ascending=False)
# )

# corr.head(30).sort_values().plot(kind="barh")

# plt.title("Top 30 Feature Correlations With Target")
# plt.xlabel("Correlation")
# plt.tight_layout()
# plt.show()

# classification_train_set  = classification_ds.iloc[ : int(len(classification_ds) * 0.8)] 
# classification_test_set = classification_ds.iloc[int(len(classification_ds) * 0.8) : ]

regression_train_Set = regression_ds.iloc[ : int(len(regression_ds) * 0.8)]
regression_test_Set =  regression_ds.iloc[int(len(regression_ds) * 0.8) : ]

for i in range(len(regression_ds.columns)): print(f'col.{i} : {regression_ds.columns[i]}')
# for i in range(len(classification_ds.columns)): print(f'col.{i} : {classification_ds.columns[i]}')

# print('last row : ', regression_ds[-1:])

# print("=== DATA VERIFICATION ===\n")
# print(f"Classification target unique: {classification_ds['target'].unique()}\n")
# print(f"Regression target sample: {regression_ds['target'].head()}\n")
# print(f"Regression target dtype: {regression_ds['target'].dtype}\n")
# print(f"Classification train shape: {classification_train_set.shape}\n")
# print(f"Regression train shape: {regression_train_Set.shape}\n")
# print(f"Classification test shape: {classification_test_set.shape}\n")
# print(f"Regression test shape: {regression_test_Set.shape}\n")


# model.train(regression_train_Set)
# lstm_pred_regression = model.predict( regression_test_Set)
# model.forecast(regression_ds)    
# print('lstm pred max : ',lstm_pred_regression.max())
# print('lstm pred min : ',lstm_pred_regression.min())

# model2.train(regression_train_Set)
# xgb_pred_regression = model2.predict(regression_test_Set)
# model2.forecast(regression_ds)
# print('xgb pred max : ',xgb_pred_regression.max())
# print('xgb pred min : ',xgb_pred_regression.min())  

# cat_boost.train(regression_train_Set)
# cat_reg_pred = cat_boost.predict(regression_test_Set)
# cat_boost.forecast(regression_ds)
# print('cat pred max : ',cat_reg_pred.max())
# print('cat pred min : ',cat_reg_pred.min())

# light_gbm.train(regression_train_Set)
# lightgbm_reg_pred = light_gbm.predict(regression_test_Set)
# light_gbm.forecast(regression_ds)
# print('lightgbm pred max : ',lightgbm_reg_pred.max())
# print('lightgbm pred min : ',lightgbm_reg_pred.min())

# tft_model = TFTQuantModel()
# tft_model.train(train_df=regression_train_Set)
# tft_reg_pred =  tft_model.predict(test_df = regression_test_Set)
# tft_model.forecast(regression_test_Set)


# lstm_clf_model.train(classification_train_set)
# lstm_clf_model.predict(classification_test_set)

# xgb_clf_model.train(classification_train_set)
# xgb_clf_model.predict(classification_test_set)
# xgb_clf_model.forecast(classification_ds)


# backtest_manager = ModelBacktester(initial_cash=10000.0, commission=0.001)
# lstm_performance = backtest_manager.run_backtest(model_name="LSTM",raw_df=data_for_backtesting,test_set=regression_test_Set,predictions=lstm_pred_regression)
# xgb_performance = backtest_manager.run_backtest(model_name="XGBoost",raw_df=data_for_backtesting,test_set=regression_test_Set,predictions=xgb_pred_regression)
# cat_performance = backtest_manager.run_backtest(model_name="CATBoost",raw_df=data_for_backtesting,test_set=regression_test_Set,predictions=cat_reg_pred)
# lightGBM_performance = backtest_manager.run_backtest(model_name="LightGBM",raw_df=data_for_backtesting,test_set=regression_test_Set,predictions=lightgbm_reg_pred)


# ******************************************************************************************************

# nifty_df = get_nifty_data()
# train_loader, val_loader, test_loader, train_dataset = prepare_tft_data(nifty_df=nifty_df)
# tft_model, lightning_trainer = initialize_tft_and_trainer(train_dataset)
# lightning_trainer.fit(tft_model,train_dataloaders=train_loader, val_dataloaders=val_loader)

# run_predictions_and_plot(best_checkpoint_path=r'D:\Stock Price Forecasting\backend\tft_logs\nifty50_forecast\version_3\checkpoints\best-tft-nifty50-epoch=05-val_loss=0.0138.ckpt',
#                          test_dataloader=test_loader,df=nifty_df)


# live_target = run_live_future_prediction(
#     best_checkpoint_path=r'D:\Stock Price Forecasting\backend\tft_logs\nifty50_forecast\version_3\checkpoints\best-tft-nifty50-epoch=05-val_loss=0.0138.ckpt', 
#     df=nifty_df, 
#     asset_ticker="RELIANCE.NS")
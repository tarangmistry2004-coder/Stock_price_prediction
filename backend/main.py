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

import uvicorn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware

lstm_model = LSTMmodel()
xgb_model = XGBoostModel()
# model3 = ProphetModel()
# lstm_clf_model = LSTM_CLFmodel()
# xgb_clf_model = XGBoost_CLFModel()
cat_boost = CatBoostModel()
light_gbm = LightGBMModel()

# classification_ds , data_for_backtesing = get_stock_data('reliance.ns' , 'max' , True)
# classification_ds.to_csv('RS_cls.csv')
# classification_train_set  = classification_ds.iloc[ : int(len(classification_ds) * 0.8)] 
# classification_test_set = classification_ds.iloc[int(len(classification_ds) * 0.8) : ]
# for i in range(len(regression_ds.columns)): print(f'col.{i} : {regression_ds.columns[i]}')
# for i in range(len(classification_ds.columns)): print(f'col.{i} : {classification_ds.columns[i]}')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],      
    allow_headers=["*"],
)

@app.get(path='/')
def home():
    return {'Message'  : 'Server is on !!','status' : 'OK'}

@app.get(path = '/get_reliance_forecating')
def reliance_forecasting():
    regression_ds , data_for_backtesting = get_stock_data('reliance.ns','max')
    regression_ds.to_csv('RS_reg.csv')
    regression_train_Set = regression_ds.iloc[ : int(len(regression_ds) * 0.8)]
    regression_test_Set =  regression_ds.iloc[int(len(regression_ds) * 0.8) : ]
    
    # LSTM
    lstm_model.train(regression_train_Set)
    lstm_pred_regression = lstm_model.predict(regression_test_Set)
    lstm_forecasting = lstm_model.forecast(regression_ds)

    # XGB
    xgb_model.train(regression_train_Set)
    xgb_pred_regression = xgb_model.predict(regression_test_Set)
    xgb_forecasting = xgb_model.forecast(regression_ds)

    # catboost
    cat_boost.train(regression_train_Set)
    cat_reg_pred = cat_boost.predict(regression_test_Set)
    catboost_forecasting = cat_boost.forecast(regression_ds)

    # light GBM
    light_gbm.train(regression_train_Set)
    lightgbm_reg_pred = light_gbm.predict(regression_test_Set)
    lightgbm_forecasting = light_gbm.forecast(regression_ds)

    final_result  = {
        'lstm' : lstm_forecasting,
        'xgb' : xgb_forecasting,
        'catboost'  : catboost_forecasting,
        'lightgbm' : lightgbm_forecasting
    }

    return final_result



# backtest_manager = ModelBacktester(initial_cash=10000.0, commission=0.001)
# lstm_performance = backtest_manager.run_backtest(model_name="LSTM",raw_df=data_for_backtesting,test_set=regression_test_Set,predictions=lstm_pred_regression)
# xgb_performance = backtest_manager.run_backtest(model_name="XGBoost",raw_df=data_for_backtesting,test_set=regression_test_Set,predictions=xgb_pred_regression)
# cat_performance = backtest_manager.run_backtest(model_name="CATBoost",raw_df=data_for_backtesting,test_set=regression_test_Set,predictions=cat_reg_pred)
# lightGBM_performance = backtest_manager.run_backtest(model_name="LightGBM",raw_df=data_for_backtesting,test_set=regression_test_Set,predictions=lightgbm_reg_pred)


# ******************************************************************************************************

# lightning_trainer.fit(tft_model,train_dataloaders=train_loader, val_dataloaders=val_loader)
# tft_model, lightning_trainer = initialize_tft_and_trainer(train_dataset)
# train_loader, val_loader, test_loader, train_dataset = prepare_tft_data(nifty_df=nifty_df)
# run_predictions_and_plot(best_checkpoint_path=r'D:\Stock Price Forecasting\backend\tft_logs\nifty50_forecast\version_3\checkpoints\best-tft-nifty50-epoch=05-val_loss=0.0138.ckpt',
#                         test_dataloader=test_loader,df=nifty_df)

@app.get('/TFT_model')
def run_tft_model(ticker_name : str = 'reliance.ns'):
    nifty_df = get_nifty_data()
    tft_result = run_live_future_prediction(
    best_checkpoint_path=r'D:\Stock Price Forecasting\backend\tft_logs\nifty50_forecast\version_3\checkpoints\best-tft-nifty50-epoch=05-val_loss=0.0138.ckpt', 
                     df=nifty_df, asset_ticker=ticker_name.upper())
    return tft_result





if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
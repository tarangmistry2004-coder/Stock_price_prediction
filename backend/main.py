from DataCollection.DataFetch import get_stock_data

from models.LSTMmodel import LSTMmodel
from models.XGBoostmodel import XGBoostModel
from models.Prophetmodel import ProphetModel
from models.LSTM_clf import LSTM_CLFmodel
from models.XGBoost_clf import XGBoost_CLFModel
from models.TFT import TFTQuantModel
from models.catmodel import CatBoostModel
from models.LightGBMmodel import LightGBMModel

from BT.test1 import run_production_backtest


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

# classification_ds = get_stock_data('reliance.ns' , 'max' , True)
regression_ds = get_stock_data('reliance.ns','max')
# regression_ds = get_stock_data('TCS.bo','max')

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

# for i in range(len(regression_ds.columns)): print(f'col.{i} : {regression_ds.columns[i]}')
# for i in range(len(classification_ds.columns)): print(f'col.{i} : {classification_ds.columns[i]}')


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

model2.train(regression_train_Set)
xgb_pred_regression = model2.predict(regression_test_Set)
model2.forecast(regression_ds)

cat_boost.train(regression_train_Set)
cat_reg_pred = cat_boost.predict(regression_test_Set)
cat_boost.forecast(regression_ds)

light_gbm.train(regression_train_Set)
lightgbm_reg_pred = light_gbm.predict(regression_test_Set)
light_gbm.forecast(regression_ds)

# tft_model = TFTQuantModel()
# tft_model.train(train_df=regression_train_Set)
# tft_reg_pred =  tft_model.predict(test_df = regression_test_Set)
# tft_model.forecast(regression_test_Set)

# run_production_backtest(xgb_pred_regression,regression_test_Set,cash=10000)

# lstm_clf_model.train(classification_train_set)
# lstm_clf_model.predict(classification_test_set)

# xgb_clf_model.train(classification_train_set)
# xgb_clf_model.predict(classification_test_set)
# xgb_clf_model.forecast(classification_ds)

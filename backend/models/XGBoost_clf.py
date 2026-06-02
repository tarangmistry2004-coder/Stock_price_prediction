from xgboost import XGBClassifier 
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap

class XGBoost_CLFModel:

    def __init__(self):
        self.model = None
        self.lookBack = 20
        self.feature_scaler = MinMaxScaler()
        self.df_columns = None
        self.target_col = 'target'

    def _prepare_sequence(self,features: np.ndarray,targets: np.ndarray):

        x, y = [], []

        for i in range(self.lookBack, len(features)):
            seq = features[i - self.lookBack + 1 : i + 1]
            # seq = features[i - self.lookBack  : i ]
            x.append(seq.flatten())
            y.append(targets[i])

        return (np.array(x, dtype=np.float32),np.array(y, dtype=np.float32))
    
    def train(self, df: pd.DataFrame):
        print('XGB classification traning ....')
        df = df.dropna().copy()

        self.df_columns = [c for c in df.columns if c != self.target_col]

        features_raw = df[self.df_columns].values
        targets_raw  = df[self.target_col].values

        features_scaled = self.feature_scaler.fit_transform(features_raw)

        x, y = self._prepare_sequence(features_scaled,targets_raw)

        n_down = np.sum(y == 0)
        n_up   = np.sum(y == 1)

        scale_pos_weight = n_down / max(n_up, 1)

        print( f"Class dist — Down: {n_down}, Up: {n_up}, scale_pos_weight: {scale_pos_weight:.3f}")

        self.model = XGBClassifier(

            objective='binary:logistic',
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
        )

        # self.model.fit(features_scaled, targets_raw)
        self.model.fit(x,y)

        # print(self.model.feature_importances_)
        # print(len(self.df_columns))
        # print(len(self.model.feature_importances_))

        # importance = pd.DataFrame({
        #     "Feature": self.df_columns,
        #     "Importance": self.model.feature_importances_
        # })

        # importance = importance.sort_values(
        #     by="Importance",
        #     ascending=False
        # )

        # print(importance)

        
    def predict(self, df: pd.DataFrame) -> pd.Series:

        if self.model is  None:
            raise RuntimeError("Call train() before predict()!")

        df = df.dropna().copy()
       
        # y_true = df[self.target_col].values[self.lookBack:]
        y_true = df[self.target_col].values

        features_scaled = self.feature_scaler.transform(df[self.df_columns].values)
        targets_raw = df[self.target_col].values

        x, y = self._prepare_sequence(features_scaled,targets_raw)
       
        # pred_proba = self.model.predict_proba(features_scaled)[:, 1]
        pred_proba = self.model.predict_proba(x)[:, 1]
        pred_label = (pred_proba >= 0.5).astype(int)

        print('\nClassifiaction result :')
        # print('cls y_true : \n',y_true[-20:] , '\n', 'cls y_pred : \n',pred_label[-20:])
        print(f"\nAccuracy : {accuracy_score(y, pred_label) * 100:.2f}%")
        print(f'Confusion matrix :\n {confusion_matrix(y,pred_label)}')
        print(f"Baseline (majority class): {max(y_true.mean(), 1-y_true.mean())*100:.2f}%")
        print(classification_report(y,pred_label,target_names=['Down', 'Up']))

      
        confident_mask = ((pred_proba > 0.60) | (pred_proba < 0.40))

        if confident_mask.sum() > 0:
            filtered_pred = np.where(pred_proba[confident_mask] >= 0.5,1,0)
            filtered_true = y[confident_mask]
            filtered_acc = accuracy_score(filtered_true,filtered_pred)
            print(f"\nFiltered Accuracy : {filtered_acc * 100:.2f}%")
            print(f"Trade Frequency : "f"{confident_mask.mean() * 100:.2f}%")

        # result = permutation_importance(
        #     self.model,
        #     features_scaled,
        #     targets_raw,
        #     n_repeats=10,
        #     random_state=42
        # )

        # importance = pd.DataFrame({
        #     "Feature": self.df_columns,
        #     "Importance": result.importances_mean
        # })

        # print(
        #     importance.sort_values(
        #         by="Importance",
        #         ascending=False
        #     )
        # )
    def forecast(self, df: pd.DataFrame):

        if  self.model is None:
            raise RuntimeError("Call train() before forecast()!")

        df_clean = df[self.df_columns].dropna().copy()
        features_scaled = self.feature_scaler.transform(df_clean[self.df_columns].values)
        last_window = features_scaled[-self.lookBack:]

        x = last_window.flatten().reshape(1, -1)
        # x = last_window[-1].reshape(1,-1)

        pred_proba = self.model.predict_proba(x)[0,1]

        direction = "UP" if pred_proba >= 0.5 else "DOWN"

        confidence = ("High" if abs(pred_proba - 0.5) > 0.15 else "Low")

        last_date = df_clean.index[-1]

        next_day = (last_date + pd.offsets.BusinessDay(1)).strftime('%Y-%m-%d')

        print("\nForecast")
        print(f"Last date    : {last_date.date()}")
        print(f"Forecast for : {next_day}")
        # print(f"Probability  : {pred_proba * 100:.2f}%")
        print(f"Direction    : {direction}")
        print(f"Confidence   : {confidence}")

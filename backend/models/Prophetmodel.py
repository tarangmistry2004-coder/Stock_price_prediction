from prophet import Prophet
import pandas as pd
import numpy as np


class ProphetModel():
    def __init__(self):
        self.forecating_days = 2
        self.model = None
        self.changepoint_scale = 0.4

    def train(self,df : pd.DataFrame):
        df = df.reset_index().rename(columns = {'Date':'ds','Close' : 'y'})[['ds','y']]
        df['ds'] = pd.to_datetime(df['ds'].dt.tz_localize(None))

        self.model  = Prophet(changepoint_prior_scale=self.changepoint_scale,
                              daily_seasonality=False,
                              weekly_seasonality=True,
                              yearly_seasonality=True)
        self.model.fit(df)
        
    def test(self , df : pd.DataFrame):
        if self.model == None:
            self.train(df)

        forecast = self.model.make_future_dataframe(self.forecating_days , freq='B')
        forecast["ds"] = forecast["ds"].dt.tz_localize(None)

        result = self.model.predict(forecast)

        result_last_row = result.tail(self.forecating_days)
        result_price = result_last_row['yhat'].tolist()
        dates = result_last_row["ds"].dt.strftime("%Y-%m-%d").tolist()

        last_close = float(df["Close"].iloc[-1])
        avg_forecast = float(np.mean(result_price))
        return_pct = (avg_forecast - last_close) / last_close
        
        print('Prophet')
        print('Dates : ' , dates)
        print('future price : ', result_price)
        print('Est. Return% :', round(return_pct * 100,2) , '%')
import pandas as pd
import yfinance as yf
import numpy as np
import ta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
_nifty50_dict = {
    'ADANIENT.NS': 'Metals & Mining',
    'ADANIPORTS.NS': 'Services',
    'APOLLOHOSP.NS': 'Healthcare',
    'ASIANPAINT.NS': 'Consumer Durables',
    'AXISBANK.NS': 'Financial Services',
    'BAJAJ-AUTO.NS': 'Automobile and Auto Components',
    'BAJFINANCE.NS': 'Financial Services',
    'BAJAJFINSV.NS': 'Financial Services',
    'BEL.NS': 'Capital Goods',
    'BHARTIARTL.NS': 'Telecommunication',
    'CIPLA.NS': 'Healthcare',
    'COALINDIA.NS': 'Oil, Gas & Consumable Fuels',
    'DRREDDY.NS': 'Healthcare',
    'EICHERMOT.NS': 'Automobile and Auto Components',
    'ETERNAL.NS': 'Consumer Services',
    'GRASIM.NS': 'Construction Materials',
    'HCLTECH.NS': 'Information Technology',
    'HDFCBANK.NS': 'Financial Services',
    'HDFCLIFE.NS': 'Financial Services',
    'HINDALCO.NS': 'Metals & Mining',
    'HINDUNILVR.NS': 'Fast Moving Consumer Goods',
    'ICICIBANK.NS': 'Financial Services',
    'INDIGO.NS': 'Services',
    'INFY.NS': 'Information Technology',
    'ITC.NS': 'Fast Moving Consumer Goods',
    'JIOFIN.NS': 'Financial Services',
    'JSWSTEEL.NS': 'Metals & Mining',
    'KOTAKBANK.NS': 'Financial Services',
    'LT.NS': 'Construction',
    'M&M.NS': 'Automobile and Auto Components',
    'MARUTI.NS': 'Automobile and Auto Components',
    'MAXHEALTH.NS': 'Healthcare',
    'NESTLEIND.NS': 'Fast Moving Consumer Goods',
    'NTPC.NS': 'Power',
    'ONGC.NS': 'Oil, Gas & Consumable Fuels',
    'POWERGRID.NS': 'Power',
    'RELIANCE.NS': 'Oil, Gas & Consumable Fuels',
    'SBILIFE.NS': 'Financial Services',
    'SHRIRAMFIN.NS': 'Financial Services',
    'SBIN.NS': 'Financial Services',
    'SUNPHARMA.NS': 'Healthcare',
    'TCS.NS': 'Information Technology',
    'TATACONSUM.NS': 'Fast Moving Consumer Goods',
    'TATAMOTORS.NS': 'Automobile and Auto Components',
    'TATASTEEL.NS': 'Metals & Mining',
    'TECHM.NS': 'Information Technology',
    'TITAN.NS': 'Consumer Durables',
    'TRENT.NS': 'Consumer Services',
    'ULTRACEMCO.NS': 'Construction Materials',
    'WIPRO.NS': 'Information Technology'
}

def engineer_features_per_ticker(group):
    group = group.sort_values('Date').copy()
    
    group['RSI'] = RSIIndicator(group['Close']).rsi()
    group['Return_1d'] = group['Close'].pct_change(1)
    group['Return_5d'] = group['Close'].pct_change(5)
    group['ATR'] = AverageTrueRange(high=group['High'],low=group['Low'],close=group['Close']).average_true_range() / group['Close']
   
    rolling_mean = group['Close'].rolling(window=20).mean()
    rolling_std = group['Close'].rolling(window=20).std()
    group['Z_Score_20'] = (group['Close'] - rolling_mean) / (rolling_std + 1e-9)
  
    group['Volume_Ratio'] = group['Volume'] / (group['Volume'].rolling(window=20).mean() + 1e-9)
    
   
    delta = group['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    group['RSI_14'] = 100 - (100 / (1 + rs))
    
    group['Target'] = group['Return_1d'].shift(-1)
    
    return group


def get_nifty_data():
    tickers = list(_nifty50_dict.keys())
    
    raw_data = yf.download(tickers, start="2020-01-01", group_by='ticker')
    
    panel_list = []
    
    for ticker in tickers:
        if ticker not in raw_data.columns.levels[0]:
            print(f"Warning: {ticker} not found")
            continue
            
        df_ticker = raw_data[ticker].copy()
        df_ticker = df_ticker.dropna(subset=['Close']) 
        
        if df_ticker.empty:
            continue
            
       
        df_ticker['Ticker'] = ticker
        df_ticker['Sector'] = _nifty50_dict[ticker]
        df_ticker = df_ticker.reset_index()
        
        df_ticker['DayOfWeek'] = df_ticker['Date'].dt.dayofweek
        df_ticker['Month'] = df_ticker['Date'].dt.month
        
        df_ticker = engineer_features_per_ticker(df_ticker)
        
        panel_list.append(df_ticker)
        
        
    master_df = pd.concat(panel_list, axis=0).reset_index(drop=True)

    processed_df = master_df.dropna().reset_index(drop=True)
    processed_df = processed_df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    processed_df['TimeIndex'] = processed_df.groupby('Ticker').cumcount()
    
    print(f"Dataset Size: {processed_df.shape}")
    print(f'top 5 row : {processed_df.head()}')
    print(f'columns : {processed_df.columns}')
    

    processed_df.to_csv("nifty50_processed.csv", index=False)

    return processed_df

# get_nifty_data()



import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime, timezone
from typing import List, Dict
from bs4 import BeautifulSoup
import feedparser

from ta.momentum import RSIIndicator , StochasticOscillator 
from ta.trend import MACD , ADXIndicator 
from ta.volatility import BollingerBands, AverageTrueRange 

from transformers import pipeline

# ── load FinBERT once at module level (not inside the function) ──────────────
_finbert = pipeline(
    'sentiment-analysis',
    model     = 'ProsusAI/finbert',
    tokenizer = 'ProsusAI/finbert',
    device    = -1,         
    batch_size= 16,
)


def _add_technical_data(df: pd.DataFrame, nifty_close: pd.Series) -> pd.DataFrame:
    data = df.copy()

    high = data['High']
    low = data['Low']
    op  = data['Open']
    close = data['Close']
    vol = data['Volume']

    # Momentum 
    data['RSI'] = RSIIndicator(close, window=14).rsi()
    data['RSI_change']  = data['RSI'].diff(1)
    data['RSI_overbought'] = (data['RSI'] > 70).astype(float)
    data['RSI_oversold'] = (data['RSI'] < 30).astype(float)

    _macd = MACD(close)
    data['MACD_diff'] = _macd.macd_diff()                       
    _macd_line  = _macd.macd()
    _macd_signal = _macd.macd_signal()
    data['MACD_cross'] = np.sign(_macd_line - _macd_signal)
    data['MACD_cross_change'] = data['MACD_cross'].diff()         

    # Volatility 
    _bb = BollingerBands(close)
    _bb_high = _bb.bollinger_hband()
    _bb_low = _bb.bollinger_lband()
    data['BB_diff'] = (_bb_high - _bb_low) / close
    data['BB_position'] = (close - _bb_low) / (_bb_high - _bb_low).replace(0, np.nan)

    data['ATR'] = AverageTrueRange(high, low, close).average_true_range() / close
    data['realized_vol_5d'] = close.pct_change().rolling(5).std()
    data['realized_vol_20d'] = close.pct_change().rolling(20).std()
    data['vol_regime'] = (data['realized_vol_5d'] / data['realized_vol_20d'].replace(0, np.nan))

    # Trend 
    _sma20 = close.rolling(20).mean()
    _sma50 = close.rolling(50).mean()
    data['close_to_SMA20'] = (close - _sma20) / _sma20.replace(0, np.nan)
    data['close_to_SMA50'] = (close - _sma50) / _sma50.replace(0, np.nan)
    data['SMA20_to_SMA50'] = (_sma20 - _sma50) / _sma50.replace(0, np.nan)

    # Mean Reversion 
    _std20 = close.rolling(20).std()
    data['zscore_20d']  = (close - _sma20) / _std20.replace(0, np.nan)
    data['dist_52w_high'] = close / close.rolling(252).max() - 1
    data['dist_52w_low'] = close / close.rolling(252).min() - 1

    # Returns 
    data['Return_1d'] = close.pct_change(1)
    data['Return_5d'] = close.pct_change(5)
    data['Return_20d'] = close.pct_change(20)
    data['lag1_return'] = data['Return_1d'].shift(1)
    data['lag2_return'] = data['Return_1d'].shift(2)

    # Gap (overnight signal) 
    data['gap']  = (op - close.shift(1)) / close.shift(1).replace(0, np.nan)
    data['gap_up'] = (data['gap'] >  0.005).astype(float)
    data['gap_down'] = (data['gap'] < -0.005).astype(float)

    # Intraday Range 
    data['HL_spread'] = (high - low) / close.replace(0, np.nan)
    data['CO_spread'] = (close - op) / op.replace(0, np.nan)

    # Volume Anomaly
    _vol_mean20 = vol.rolling(20).mean()
    _vol_std20 = vol.rolling(20).std()
    data['vol_ratio'] = vol / _vol_mean20.replace(0, np.nan)
    data['vol_zscore']  = (vol - _vol_mean20) / _vol_std20.replace(0, np.nan)
    data['vol_up_confirm'] = ((data['vol_zscore'] > 2) & (data['Return_1d'] > 0)).astype(float)
    data['vol_down_confirm'] = ((data['vol_zscore'] > 2) & (data['Return_1d'] < 0)).astype(float)
    data['vol_price_confirm']= data['Return_1d'] * data['vol_ratio']

    # # Consecutive Days 
    # up_days = data['Return_1d'] > 0
    # down_days = data['Return_1d'] < 0
    # # group consecutive runs and cumcount within each run
    # data['consec_up'] = (up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()).astype(float)
    # data['consec_down'] = (down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()).astype(float)

    # Market Context — Nifty50 
    nifty_ret = nifty_close.pct_change()
    nifty_ret.index = pd.to_datetime(nifty_ret.index)
    data['nifty_ret'] = nifty_ret.reindex(data.index).ffill().fillna(0.0)
    data['rel_nifty'] = data['Return_1d'] - data['nifty_ret']

    # Calendar 
    # data['day_of_week'] = data.index.dayofweek.astype(float)   # 0=Mon, 4=Fri

    _adx_obj = ADXIndicator(high, low, close, window=14)
    data["ADX"] = _adx_obj.adx()
    data["ADX_slope"] = data["ADX"].diff(3)

    _stoch_obj = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    data["Stochastic_K"] = _stoch_obj.stoch()
    data["Stochastic_D"] = _stoch_obj.stoch_signal()

    return data


def _get_news_data(stock: str) -> List[Dict[str, str]]:
    news_data: List[Dict[str, str]] = []

    # yfinance news
    try:
        raw_news = yf.Ticker(stock).news or []
        for i in raw_news[:60]:
            content  = i.get('content', {})
            title    = content.get('title', '') or i.get('title', '')
            pub_date = content.get('pubdate', '') or i.get('providerPublishTime', '')

            if isinstance(pub_date, (int, float)):
                pub_date = datetime.fromtimestamp(pub_date, tz=timezone.utc).strftime('%Y-%m-%d')
            elif pub_date:
                pub_date = str(pub_date)[:10]
            else:
                pub_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

            if title:
                news_data.append({'title': title, 'pubdate': pub_date, 'source': 'yfinance'})

    except Exception as e:
        print(f'[WARNING] yfinance news fetch failed: {e}')   # non-fatal

    # Google RSS news
    try:
        url  = f'https://news.google.com/rss/search?q={stock}+stock&hl=en-US&gl=US&ceid=US:en'
        feed = feedparser.parse(url)
        for i in feed.entries[:60]:
            published = i.get('published', '')
            if published:
                try:
                    published = datetime(*i.published_parsed[:3]).strftime('%Y-%m-%d')
                except Exception:
                    published = published[:10]
            else:
                published = datetime.now(timezone.utc).strftime('%Y-%m-%d')

            title = BeautifulSoup(i.get('title', ''), 'lxml').get_text().strip()
            news_data.append({'title': title, 'pubdate': published, 'source': 'Google RSS'})

    except Exception as e:
        print(f'[WARNING] Google RSS fetch failed: {e}')      

    return news_data


def _news_sentiment(raw_news: List[Dict[str, str]]) -> List[Dict]:
    _label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}

    if not raw_news:
        return []

    titles  = [i['title']  for i in raw_news]
    pubdates= [i['pubdate'] for i in raw_news]

    # batch inference 
    results = _finbert(titles, truncation=True, max_length=512)

    sentiments = []
    for title, pubdate, res in zip(titles, pubdates, results):
        label = res['label'].lower()
        score = res['score']
        sentiments.append({
            'title'           : title,
            'pubdate'         : pubdate,
            'senti_confidence': score,
            'senti_label'     : label,
            'senti_score'     : _label_map.get(label, 0.0) * score,
        })

    return sentiments


def _handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    neutral_cols = [
        'sentiment',
        'sentiment_EMA',
        'vol_up_confirm',
        'vol_down_confirm',
        'gap_up',
        'gap_down',
    ]
    for col in neutral_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0.0)

    technical_cols = [
        'RSI',
        'RSI_change',
        'MACD_diff',
        'MACD_cross',
        'MACD_cross_change',
        'BB_diff',
        'BB_position',
        'realized_vol_5d',
        'realized_vol_20d',
        'vol_regime',
        'close_to_SMA20',
        'close_to_SMA50',
        'SMA20_to_SMA50',
        'zscore_20d',
        'dist_52w_high',
        'dist_52w_low',
        'Return_5d',
        'Return_20d',
        'lag1_return',
        'lag2_return',
        'gap',
        'CO_spread',
        'HL_spread',
        'ATR',
        'vol_ratio',
        'vol_zscore',
        'vol_price_confirm',
        'nifty_ret',
        'rel_nifty',
        'day_of_week',
    ]
    technical_cols = [col for col in technical_cols if col in data.columns]
    if technical_cols:
        data[technical_cols] = data[technical_cols].ffill()
        data.dropna(subset=technical_cols, inplace=True)

    
    # data.dropna(subset=['target'], inplace=True)

    return data




def get_stock_data(stock: str, period: str , for_clf : bool = False) -> pd.DataFrame:
    try:
        # price data 
        data: pd.DataFrame = yf.download(tickers=stock, period=period, interval='1d')
        if data.empty:
            raise ValueError(f'No data found for {stock}')

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.index = pd.to_datetime(data.index)
        data.dropna(inplace=True)

        data_for_backtesting = data.copy()

        # nifty50 for market context 
        nifty_raw = yf.download('^NSEI', period=period, interval='1d')
        if isinstance(nifty_raw.columns, pd.MultiIndex):
            nifty_raw.columns = nifty_raw.columns.get_level_values(0)
        nifty_close = nifty_raw['Close'].squeeze()
        nifty_close.index = pd.to_datetime(nifty_close.index)

        # technical features 
        data = _add_technical_data(data, nifty_close)

        #  sentiment
        news_data = _get_news_data(stock)
        scores    = _news_sentiment(news_data)

        if scores:
            senti_df = pd.DataFrame(scores)
            senti_df['pubdate'] = pd.to_datetime(senti_df['pubdate'])
            senti_df.sort_values('pubdate', inplace=True)
            senti_df['weighted'] = senti_df['senti_score'] * senti_df['senti_confidence']

            senti_series = (
                senti_df.groupby('pubdate')
                .apply(
                    lambda g: g['weighted'].sum() / g['senti_confidence'].sum(),
                    include_groups=False
                )
                .rename('sentiment')
            )
            senti_series = senti_series.clip(-1.0, 1.0)   # fixed: was -0.1
        else:
            senti_series = pd.Series(dtype=float, name='sentiment')

        data = data.join(senti_series, how='left')
        data['sentiment']     = data['sentiment'].fillna(0.0).shift(1)
        data['sentiment_EMA'] = data['sentiment'].ewm(3, adjust=False).mean()
        data['sentiment_EMA'] = data['sentiment_EMA'].fillna(0.0)


        cols_to_drop = [
            'High', 'Low', 'Open', 'Volume',        
            # 'MACD', 
            # 'MACD_signal',                 
            'BB_high', 'BB_low',                     
            # 'SMA20', 
            # 'SMA50',                        
            # 'EMA12', 
            # 'EMA26',                        
            'OBV',                                   
            # 'vol_SMA20', 
            # 'vol_change',                           
            # 'RSI_overbought',
            # 'RSI_oversold',
            # 'close_to_SMA50',
            # 'lag1_return',
            # 'lag2_return',
            'gap_up',
            'gap_down',
            # 'consec_up',
            # 'consec_down',
            'BB_position',
            # 'SMA20_to_SMA50',
            # # 'gap',
            # 'vol_zscore',
            # # 'ATR',
            # # 'HL_spread',
            # # 'day_of_week',
            'MACD_cross_change',
            'vol_down_confirm', 
            # 'Return_1d',   
        ]

        if for_clf:
             # target (classification)  
            ret = data['Return_1d'].shift(-1)
            data['target'] = np.where(ret >  0,  1, np.where(ret < -0,  0, np.nan))
            data.drop(columns=[c for c in cols_to_drop if c in data.columns], inplace=True)
        else :
            # target (Regression)
            data['target_vol'] = data['Return_1d'].rolling(window=20).std()
            data['target'] = (data['Return_1d'] / data['target_vol'].replace(0, np.nan)).shift(-1)


            # data['target'] = data['Return_1d'].shift(-1)
            # data['target'] = (data['Close'].shift(-5) / data['Close']) - 1
            data.drop(columns=[c for c in cols_to_drop  if c in data.columns],inplace=True)

        data = _handle_missing_values(data)

        return data , data_for_backtesting

    except Exception as e:
        raise e
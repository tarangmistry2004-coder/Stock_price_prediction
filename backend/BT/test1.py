from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ta.volatility import AverageTrueRange

class MLZScoreStrategy(Strategy):
    buy_threshold = 2.0   
    sell_threshold = -1.0 
    atr_multiplier = 2.0  

    def init(self):
        self.zscore_signal = self.I(lambda x: x, self.data.active_zscore)
        self.atr = self.I(lambda x: x, self.data.ATR) 

    def next(self):
        current_zscore = self.zscore_signal[-1]
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1] 

        if np.isnan(current_atr) or current_zscore == 0:
            return

        if self.position:
            if current_zscore <= self.sell_threshold:
                self.position.close()
            
            
            else:
                for trade in self.trades:
                    if trade.is_long:
                        potential_sl = current_price * (1 - (self.atr_multiplier * current_atr))
                        if trade.sl is None or potential_sl > trade.sl:
                            trade.sl = potential_sl
        else:
            if current_zscore >= self.buy_threshold:
                stop_loss = current_price * (1 - (self.atr_multiplier * current_atr))
                take_profit = current_price * (1 + (self.atr_multiplier * 2.5 * current_atr))
                
                self.buy(sl=stop_loss, tp=take_profit)

def plot_charts(stats, model_name="Machine Learning Model"):
   
    equity_df = stats['_equity_curve']
    trades_df = stats['_trades']
    
   
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=False)
    plt.subplots_adjust(hspace=0.4)
    
  
    #  1 The Equity Curve
 
    axes[0].plot(equity_df.index, equity_df['Equity'], color='#2ca02c', label='Strategy Equity', linewidth=2)
    axes[0].set_title(f'{model_name} - Compounding Equity Curve', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper left')
    
 
    # 2 The Drawdown Profile (Underwater Chart)
   
    # Convert drawdown decimal to percentage
    drawdown_pct = equity_df['DrawdownPct'] * -100 
    
    axes[1].fill_between(equity_df.index, drawdown_pct, 0, color='#d62728', alpha=0.3, label='Drawdown %')
    axes[1].plot(equity_df.index, drawdown_pct, color='#d62728', linewidth=1)
    axes[1].set_title('Drawdown Profile (Capital Risk)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_ylim(drawdown_pct.min() * 1.2, 0) # Dynamic scaling
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
   
    #  3 Trade Return Distribution
   
    if not trades_df.empty:
        # Multiply by 100 to convert decimal returns to percentages
        trade_returns = trades_df['ReturnPct'] * 100
        
        sns.histplot(trade_returns, kde=True, ax=axes[2], color='#1f77b4', bins=15)
        axes[2].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Break-Even')
        axes[2].axvline(trade_returns.mean(), color='orange', linestyle='-', linewidth=1.5, 
                        label=f'Avg Trade: {trade_returns.mean():.2f}%')
        
        axes[2].set_title('Distribution of Individual Trade Returns', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Trade Return (%)', fontsize=12)
        axes[2].set_ylabel('Frequency (Count)', fontsize=12)
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'No trades executed to display distribution.', ha='center', va='center')

    # plt.show()



class ModelBacktester:
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.001):
        self.initial_cash = initial_cash
        self.commission = commission 

    def run_backtest(self, model_name: str, raw_df: pd.DataFrame, test_set: pd.DataFrame, predictions: np.ndarray) -> pd.Series:
       
        df_bt = raw_df.copy()
        
        atr_indicator = AverageTrueRange(
            high=df_bt['High'], 
            low=df_bt['Low'], 
            close=df_bt['Close']
        )
       
        df_bt['ATR'] = atr_indicator.average_true_range() / df_bt['Close'] 
        
        raw_pred_col = f"{model_name.lower()}_pred"
        df_bt[raw_pred_col] = np.nan
        df_bt.loc[test_set.index, raw_pred_col] = predictions
        
        window = 100
        rolling_mean = df_bt[raw_pred_col].rolling(window=window, min_periods=window).mean()
        rolling_std = df_bt[raw_pred_col].rolling(window=window, min_periods=window).std()
        
        zscore_col = f"{model_name.lower()}_zscore"
        df_bt[zscore_col] = (df_bt[raw_pred_col] - rolling_mean) / rolling_std
        df_bt[zscore_col] = df_bt[zscore_col].replace([np.inf, -np.inf], 0).fillna(0)
        
       
        test_period_df = df_bt.loc[test_set.index].copy()
        test_period_df['active_zscore'] = test_period_df[zscore_col]
        
      
        bt = Backtest(test_period_df, MLZScoreStrategy, cash=10000, commission=0.0005, exclusive_orders=True)
        
        stats = bt.run()

        print(f'\nMODEL : {model_name}')
        print(stats)
        plot_charts(stats , model_name=model_name)
        return stats
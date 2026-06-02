from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd


def _prepare_bt_data(test_df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Prepares and aligns the input feature data for the backtesting engine.

    Ensures that trailing metrics are shifted correctly to prevent look-ahead
    bias.
    """
    df = test_df.copy()

    # Reconstruct structural price bars if missing from the raw input series
    if "Open" not in df.columns:
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    if "High" not in df.columns:
        df["High"] = df["Close"]
    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0

    # CRITICAL LEAKAGE FIX: Shift the model predictions forward by exactly 1 index step.
    # This guarantees that the prediction generated at Wednesday's close is executed
    # at Thursday's opening bell, completely neutralizing look-ahead leakage.
    aligned_signals = pred.reindex(df.index).shift(1)

    # Capture your engineered ATR feature to use for volatility stop-losses
    atr_series = df["ATR"] if "ATR" in df.columns else df["Close"] * 0.011

    bt_df = pd.DataFrame(
        {
            "Open": df["Open"],
            "High": df["High"],
            "Low": df["Low"],
            "Close": df["Close"],
            "Volume": df["Volume"],
            "signal": aligned_signals,
            "atr_signal": atr_series,
        }
    ).dropna(subset=["Close", "signal"])

    bt_df.index = pd.to_datetime(bt_df.index)
    bt_df.sort_index(inplace=True)
    return bt_df


class ProductionAlphaReversalStrategy(Strategy):
    """Production-optimized intraday trading strategy built around your model.

    Executes trades at the opening bell based on the prior session's forecast,
    manages risk using an ATR trailing stop-loss, and programmatically closes
    positions at the closing bell.
    """

    # Hyperparameters for optimization
    threshold = 0.0010  # 0.10% expected return threshold to trigger a trade
    atr_multiplier = 1.50  # Risk multiple used to anchor stop-losses

    def init(self):
        # Bind the data arrays to the strategy class
        self.pred_return = self.I(
            lambda: self.data.signal, name="Model_Forecast"
        )
        self.atr_signal = self.I(
            lambda: self.data.atr_signal, name="ATR_Volatility"
        )

    def next(self):
        # Extract the most recent signal vector
        forecast = self.pred_return[-1]
        current_atr = self.atr_signal[-1]
        current_price = self.data.Close[-1]

        if np.isnan(forecast):
            return

        # ─── RULE 1: INTRADAY HORIZON TIME-EXIT ───
        # Your target matrix is built on a 1-day shift boundary.
        # Force-close any open positions from the prior session to prevent overnight gap risk.
        if self.position:
            self.position.close()

        # ─── RULE 2: DIRECTIONAL ENTRY REGIME ───
        if forecast > self.threshold:
            # Predicted positive move: Execute a long position at the market open
            # Secure the position using a strict volatility-adjusted stop-loss
            stop_price = current_price - (self.atr_multiplier * current_atr)
            self.buy(sl=stop_price)

        elif forecast < -self.threshold:
            # Predicted negative move: Execute a short position at the market open
            stop_price = current_price + (self.atr_multiplier * current_atr)
            self.sell(sl=stop_price)


def run_production_backtest(
    pred_return: pd.Series,
    test_df: pd.DataFrame,
    cash: float = 100_000,
    commission: float = 0.0002,  # Covers standard institutional exchange fees
    optimize: bool = True,
) -> dict:
    """Initializes and runs the backtesting simulation loop."""
    bt_df = _prepare_bt_data(test_df, pred_return)

    bt = Backtest(
        bt_df,
        ProductionAlphaReversalStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True,  # Automatically cancels opposing open orders
    )

    if optimize:
        print("\n── Optimizing System Trading Parameters ──")
        opt_stats = bt.optimize(
            threshold=[0.0005, 0.001, 0.0015, 0.002, 0.003],
            atr_multiplier=[1.0, 1.2, 1.5, 1.8, 2.0],
            maximize="Sharpe Ratio",
            return_heatmap=False,
        )
        print(f"  Optimal Signal Threshold : {opt_stats._strategy.threshold}")
        print(
            f"  Optimal ATR Risk Multiple: {opt_stats._strategy.atr_multiplier}"
        )
        stats = opt_stats
    else:
        stats = bt.run()

    # ─── FIXED CRITICAL KEY MATCHING DICTIONARY ──────────────────────────────
    # Using dynamic .get() handles key variations across library updates safely [1.1]
    final_equity = stats.get("Equity Final [$]", stats.get("Equity Final", cash))
    total_return = stats.get("Return [%]", stats.get("Return %", 0.0))
    bh_return = stats.get("Buy & Hold Return [%]", stats.get("Buy & Hold Return %", 0.0))
    max_drawdown = stats.get("Max. Drawdown [%]", stats.get("Max. Drawdown %", 0.0))
    sharpe_ratio = stats.get("Sharpe Ratio", 0.0)
    total_trades = stats.get("# Trades", 0)
    win_rate = stats.get("Win Rate [%]", stats.get("Win Rate %", 0.0))

    # Log metrics out to the terminal console
    print("\n" + "=" * 15 + " BACKTEST OUTPUT SUMMARY " + "=" * 15)
    print(f"Start Balance            : ₹{cash:,.2f}")
    print(f"Final Equity Account     : ₹{final_equity:,.2f}")
    print(f"Total Return Metric      : {total_return:.2f}%")
    print(f"Buy & Hold Return        : {bh_return:.2f}%")
    print(f"Max Peak Drawdown        : {max_drawdown:.2f}%")
    print(f"Sharpe Ratio Metric      : {sharpe_ratio:.4f}")
    print(f"Total Closed Trades Executed: {total_trades}")
    print(f"Win Rate Percentage      : {win_rate:.2f}%")
    print("=" * 55 + "\n")

    # Save the interactive chart file securely
    try:
        bt.plot(filename="production_backtest_report.html", open_browser=False)
        print(
            "  Interactive chart report saved → production_backtest_report.html"
        )
    except Exception as e:
        print(f"[WARNING] Chart plotting skipped: {e}")

    return stats

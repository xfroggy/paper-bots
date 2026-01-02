import argparse
import os
from dataclasses import dataclass
import pandas as pd
import yfinance as yf

@dataclass
class Config:
    symbol: str
    start: str
    ma_period: int
    initial_capitals: list[float]
    out_dir: str

def fetch_data(symbol: str, start: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for symbol={symbol}.")
    if "Close" not in df.columns:
        raise ValueError("Downloaded data missing Close column.")
    return df

def add_signals(df: pd.DataFrame, ma_period: int) -> pd.DataFrame:
    df = df.copy()
    df["ma"] = df["Close"].rolling(ma_period).mean()
    df = df.dropna()

    # Trend rule (daily):
    # Buy when Close crosses ABOVE MA (0->1), Sell when Close crosses BELOW MA (1->0)
    df["in_trend"] = (df["Close"] > df["ma"]).astype(int)
    df["in_trend_prev"] = df["in_trend"].shift(1).fillna(0).astype(int)

    df["entry"] = (df["in_trend_prev"] == 0) & (df["in_trend"] == 1)
    df["exit"]  = (df["in_trend_prev"] == 1) & (df["in_trend"] == 0)
    return df

def latest_signal(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (signal, date_str) where signal in {"BUY","SELL","HOLD"} based on latest row.
    """
    last = df.iloc[-1]
    date_str = str(df.index[-1].date())

    if bool(last["entry"]):
        return "BUY", date_str
    if bool(last["exit"]):
        return "SELL", date_str
    return "HOLD", date_str

def backtest(df: pd.DataFrame, initial_cash: float) -> tuple[float, float, pd.DataFrame, pd.Series]:
    cash = float(initial_cash)
    units = 0.0  # shares for stocks, coin units for crypto
    trades = []
    equity_curve = []

    for ts, row in df.iterrows():
        price = float(row["Close"])

        # Execute at close (paper simulation)
        if bool(row["entry"]) and units == 0 and price > 0:
            units = cash / price
            cash = 0.0
            trades.append({"date": str(ts.date()), "action": "BUY", "price": price, "units": units})

        elif bool(row["exit"]) and units > 0 and price > 0:
            cash = units * price
            trades.append({"date": str(ts.date()), "action": "SELL", "price": price, "units": units})
            units = 0.0

        equity_curve.append((ts, cash + units * price))

    equity = pd.Series([v for _, v in equity_curve], index=[t for t, _ in equity_curve], name="equity")
    final_value = float(equity.iloc[-1])
    roi_pct = ((final_value - initial_cash) / initial_cash) * 100.0

    trades_df = pd.DataFrame(trades)
    return final_value, roi_pct, trades_df, equity

def max_drawdown_pct(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min() * 100.0)  # negative number

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_latest_signal(out_dir: str, symbol: str, signal: str, date_str: str) -> str:
    """
    Writes a simple text file used by GitHub Actions to decide whether to send email.
    Returns the filepath.
    """
    safe_symbol = symbol.replace("-", "_")
    path = os.path.join(out_dir, f"{safe_symbol}_latest_signal.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"symbol={symbol}\n")
        f.write(f"date={date_str}\n")
        f.write(f"signal={signal}\n")
    return path

def main():
    parser = argparse.ArgumentParser(description="Daily paper trading/backtest bot (trend following).")
    parser.add_argument("--market", choices=["stocks", "crypto"], default="stocks",
                        help="Preset symbol: stocks=SPY, crypto=BTC-USD unless --symbol is set.")
    parser.add_argument("--symbol", default=None, help="Override symbol (e.g., SPY, QQQ, BTC-USD, ETH-USD).")
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--ma", type=int, default=50, help="Moving average period.")
    parser.add_argument("--capitals", default="100,1000", help="Comma-separated starting capitals.")
    parser.add_argument("--out", default="output", help="Output folder for CSV files.")
    args = parser.parse_args()

    symbol = args.symbol if args.symbol else ("SPY" if args.market == "stocks" else "BTC-USD")
    capitals = [float(x.strip()) for x in args.capitals.split(",") if x.strip()]

    cfg = Config(
        symbol=symbol,
        start=args.start,
        ma_period=args.ma,
        initial_capitals=capitals,
        out_dir=args.out
    )

    ensure_out_dir(cfg.out_dir)

    df = fetch_data(cfg.symbol, cfg.start)
    df = add_signals(df, cfg.ma_period)

    sig, sig_date = latest_signal(df)
    sig_path = write_latest_signal(cfg.out_dir, cfg.symbol, sig, sig_date)

    print(f"\nSymbol: {cfg.symbol}")
    print(f"Start: {cfg.start} | MA: {cfg.ma_period} | Bars: {len(df)}")
    print(f"Latest Signal: {sig} on {sig_date}")
    print(f"Signal file: {sig_path}")
    print(f"Capitals: {cfg.initial_capitals}\n")

    summary_rows = []

    for cap in cfg.initial_capitals:
        final_value, roi_pct, trades_df, equity = backtest(df, cap)
        mdd = max_drawdown_pct(equity)

        base = f"{cfg.symbol.replace('-', '_')}_ma{cfg.ma_period}_cap{int(cap)}"
        trades_path = os.path.join(cfg.out_dir, f"{base}_trades.csv")
        equity_path = os.path.join(cfg.out_dir, f"{base}_equity.csv")

        trades_df.to_csv(trades_path, index=False)
        equity.reset_index().rename(columns={"index": "date"}).to_csv(equity_path, index=False)

        summary_rows.append({
            "symbol": cfg.symbol,
            "start": cfg.start,
            "ma_period": cfg.ma_period,
            "capital": cap,
            "final_value": final_value,
            "roi_pct": roi_pct,
            "max_drawdown_pct": mdd,
            "num_trades": int(len(trades_df))
        })

        print(f"== Capital ${cap:,.0f} ==")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"ROI: {roi_pct:.2f}%")
        print(f"Max Drawdown: {mdd:.2f}%")
        print(f"Trades: {len(trades_df)}")
        print(f"Saved: {trades_path}")
        print(f"Saved: {equity_path}\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(cfg.out_dir, f"{cfg.symbol.replace('-', '_')}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Summary saved:", summary_path)

if __name__ == "__main__":
    main()

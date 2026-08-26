"""Data fetching and window serialization for the Kronos RAG project.

Downloads minute bars for a list of tickers from Alpaca, converts them to the
6-channel Kronos feature set (open, high, low, close, volume, amount) with
5 time features, and writes normalized sliding windows to ArrayRecord files
for the Grain dataloaders in dataset.py.

Extracted from kronos_rag.ipynb. Requires ALPACA_API_KEY / ALPACA_SECRET_KEY
in the environment.
"""

import argparse
import os
import pickle
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from array_record.python.array_record_module import ArrayRecordWriter

NY_TZ = ZoneInfo("America/New_York")

FEATURES = ("open", "high", "low", "close", "vol", "amt")

SP_TICKERS = [
    'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI',
    'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG',
    'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN',
    'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH',
    'APO', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP',
    'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF.B',
    'BG', 'BIIB', 'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK.B',
    'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT',
    'CB', 'CBOE', 'CBRE', 'CCI', 'CCJ', 'CDNS', 'CDW', 'CE', 'CEG', 'CF',
    'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA',
    'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR',
    'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSGP', 'CSX',
    'CTAS', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD',
    'DE', 'DELL', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR',
    'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN',
    'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV',
    'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS',
    'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR',
    'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO',
    'FIS', 'FITB', 'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD',
    'GDDY', 'GE', 'GEHC', 'GEV', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GM',
    'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS',
    'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE',
    'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE',
    'IDXX', 'IEX', 'IFF', 'INCY', 'INSP', 'INTC', 'INTU', 'INVH', 'IP', 'IPG',
    'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL',
    'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC',
    'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS',
    'LEN', 'LH', 'LHX', 'LRCX', 'LNC', 'LNT', 'LOW', 'LULU', 'LUV',
    'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP',
    'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX',
    'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRNA',
    'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN',
    'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC',
    'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL',
    'OKE', 'OMC', 'ON', 'ORLY', 'ORCL', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYX',
    'PAYC', 'PYPL', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR',
    'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD',
    'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'QCOM',
    'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL',
    'ROP', 'ROST', 'RSG', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB',
    'SMCI', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT',
    'STX', 'STZ', 'SWK', 'SWKS', 'SYK', 'SYF', 'SYY', 'T', 'TAP', 'TDG',
    'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS',
    'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TYL', 'UA',
    'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
    'V', 'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR',
    'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC',
    'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WTW', 'WY', 'WYNN', 'XEL',
    'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS'
]

TRAIN_FRAC = 0.80


def stamps_from_index(index):
    """Convert a DatetimeIndex to the 5 Kronos time features."""
    idx = pd.DatetimeIndex(index)
    return np.stack([
        idx.minute.values,
        idx.hour.values,
        idx.weekday.values,
        idx.day.values,
        idx.month.values,
    ], axis=-1).astype(np.int32)


def fetch_tickers(symbols, start, end, chunk=100):
    """Download minute bars for `symbols` from Alpaca.

    Returns a dict {symbol: DataFrame} with columns
    open/high/low/close/vol/amt indexed by NY-localized timestamps.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")

    client = StockHistoricalDataClient(api_key, secret_key)

    frames = {}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        req = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed="iex",
            adjustment=Adjustment.ALL
        )
        df = client.get_stock_bars(req).df
        for sym in batch:
            if isinstance(df.index, pd.MultiIndex):
                if sym not in df.index.get_level_values("symbol"):
                    continue
                sub = df.xs(sym, level="symbol").sort_index()
            else:
                sub = df.sort_index()

            sub.index = pd.DatetimeIndex(sub.index).tz_convert(NY_TZ).tz_localize(None)
            out = pd.DataFrame(index=sub.index)
            out["open"], out["high"] = sub["open"], sub["high"]
            out["low"], out["close"] = sub["low"], sub["close"]
            out["vol"] = sub["volume"]
            vwap = sub["vwap"]
            out["amt"] = sub["volume"] * vwap
            out = out.interpolate(limit_direction="both")
            frames[sym] = out
        print(f"[fetch] {min(i + chunk, len(symbols))}/{len(symbols)} symbols completed")
    return frames


def write_windows_arrayrecord(frames, train_frac, lookback, predict, output_dir, clip=5.0):
    """Split each ticker into non-overlapping lookback+predict windows.

    Every window is z-scored using the mean/std of its `lookback` past rows
    (no look-ahead leakage into the normalization) and clipped to +-clip.
    The first train_frac of each ticker's windows go to train.arrayrecord,
    the rest to val.arrayrecord.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_writer = ArrayRecordWriter(os.path.join(output_dir, "train.arrayrecord"), options="group_size:1")
    val_writer = ArrayRecordWriter(os.path.join(output_dir, "val.arrayrecord"), options="group_size:1")

    train_count, val_count = 0, 0
    window_len = lookback + predict

    for symbol, df in frames.items():
        series = df[list(FEATURES)].to_numpy(dtype=np.float32)
        stamps = stamps_from_index(df.index)
        n = len(series)

        if n < window_len:
            continue

        split_idx = int(n * train_frac)

        for i in range(0, n - window_len + 1, window_len):
            x_win = series[i : i + window_len].copy()
            st_win = stamps[i : i + window_len].copy()

            past = x_win[:lookback]
            mean = np.mean(past, axis=0, keepdims=True)
            std = np.std(past, axis=0, keepdims=True)
            x_norm = np.clip((x_win - mean) / (std + 1e-5), -clip, clip).astype(np.float32)

            record = {
                "symbol": symbol,
                "x": x_norm,
                "stamps": st_win,
            }
            serialized = pickle.dumps(record)

            if (i + window_len) <= split_idx:
                train_writer.write(serialized)
                train_count += 1
            else:
                val_writer.write(serialized)
                val_count += 1

    train_writer.close()
    val_writer.close()
    print(f"[write] Wrote {train_count} train records and {val_count} val records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ticker bars from Alpaca and save them.")
    parser.add_argument("--tickers", nargs="+", default=SP_TICKERS)
    parser.add_argument("--years", type=float, default=2)
    parser.add_argument("--out", default="frames.pkl")
    args = parser.parse_args()

    end = datetime.now(NY_TZ)
    start = end - timedelta(days=int(args.years * 365.25))
    frames = fetch_tickers(args.tickers, start, end)
    with open(args.out, "wb") as f:
        pickle.dump(frames, f)
    print(f"[fetch] Saved {len(frames)} tickers to {args.out}")

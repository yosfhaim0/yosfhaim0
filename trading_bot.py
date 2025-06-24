# Stock Trading Bot Script
# Requires: pandas, numpy, yfinance, ib_insync, scipy

import argparse
import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from ib_insync import IB, Stock
from scipy.signal import argrelextrema


def load_tickers(ticker_source):
    """Load tickers from a comma-separated string or a file."""
    if os.path.isfile(ticker_source):
        with open(ticker_source, 'r') as f:
            return [line.strip().upper() for line in f if line.strip()]
    return [t.strip().upper() for t in ticker_source.split(',') if t.strip()]


def filter_tickers(tickers):
    """Filter tickers by price and volume using yfinance."""
    print("Downloading market data for filtering...")
    data = yf.download(tickers, period="1d", threads=True, group_by='ticker', progress=False)
    filtered = []
    for t in tickers:
        try:
            info = data[t]
            price = float(info['Close'][-1])
            volume = float(info['Volume'][-1])
            if 20 <= price <= 800 and volume > 10_000_000:
                filtered.append(t)
        except Exception:
            continue
    print(f"Filtered to {len(filtered)} tickers")
    return filtered


def load_iv_cache(cache_file):
    if cache_file and os.path.isfile(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}


def save_iv_cache(cache, cache_file):
    if cache_file:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)


def fetch_iv(ticker, ib, cache, delay):
    if ticker in cache:
        return cache[ticker]
    contract = Stock(ticker, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    ticker_data = ib.reqMktData(contract, '', False, False)
    time.sleep(delay)
    iv = ticker_data.modelGreeks.impliedVolatility
    ib.cancelMktData(contract)
    if iv:
        cache[ticker] = float(iv)
    return iv


def compute_stochastic(ticker, period='6mo', k_window=14, d_window=3):
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        return None, None
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()
    k = (data['Close'] - low_min) / (high_max - low_min) * 100
    d = k.rolling(window=d_window).mean()
    return k.iloc[-1], d.iloc[-1]


def compute_support_resistance(ticker, period='6mo', order=5):
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        return None, None
    prices = data['Close']
    max_idx = argrelextrema(prices.values, np.greater, order=order)[0]
    min_idx = argrelextrema(prices.values, np.less, order=order)[0]
    resistances = prices.iloc[max_idx]
    supports = prices.iloc[min_idx]
    return supports.min() if not supports.empty else None, resistances.max() if not resistances.empty else None


def analyze_tickers(tickers, cache_file=None, delay=2.0):
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)
    except Exception as e:
        print(f"IB connection failed: {e}")
        return pd.DataFrame()
    cache = load_iv_cache(cache_file)
    results = []
    for t in tickers:
        try:
            iv = fetch_iv(t, ib, cache, delay)
            if iv is None:
                print(f"No IV for {t}")
                continue
            action = 'SELL' if iv > 0.5 else 'BUY'
            k, d = compute_stochastic(t)
            if k is None:
                continue
            signal = None
            support = resistance = None
            if k > 80 and d > 80:
                signal = 'Overbought'
                _, resistance = compute_support_resistance(t)
            elif k < 20 and d < 20:
                signal = 'Oversold'
                support, _ = compute_support_resistance(t)
            results.append({
                'Symbol': t,
                'IV': iv,
                'Action': action,
                'Signal': signal,
                '%K': k,
                '%D': d,
                'Support': support,
                'Resistance': resistance,
            })
        except Exception as e:
            print(f"Error processing {t}: {e}")
    save_iv_cache(cache, cache_file)
    ib.disconnect()
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Stock Scanner")
    parser.add_argument('--tickers', required=True, help='Comma list or file of tickers')
    parser.add_argument('--cache', default='iv_cache.json', help='Cache file for IV data')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between IB requests')
    parser.add_argument('--csv', default='results.csv', help='Output CSV file')
    parser.add_argument('--excel', default=None, help='Output Excel file')
    args = parser.parse_args()

    tickers = load_tickers(args.tickers)
    filtered = filter_tickers(tickers)
    df = analyze_tickers(filtered, cache_file=args.cache, delay=args.delay)
    if not df.empty:
        df.to_csv(args.csv, index=False)
        if args.excel:
            df.to_excel(args.excel, index=False)
        print(df)
    else:
        print("No data produced.")


if __name__ == '__main__':
    main()

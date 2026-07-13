import yfinance as yf
import numpy as np

df = yf.Ticker("RECLTD.NS").history(period="2y")

trading_days = np.array(df.index.strftime('%Y-%m-%d'))
prices       = np.round(df['Close'].values, 2)

pri = []

print(prices)
for p in prices:
    pri.append(p)

print("")
print(pri)
print(len(trading_days), len(prices))
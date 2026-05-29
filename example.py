import os
import functools

import duonlabs
import numpy as np

MODEL = "best"

## How to Forecast a single pair ##

client = duonlabs.DuonLabs(token=os.environ["DUONLABS_TOKEN"], base_url=os.environ.get("DUONLABS_API_URL"))

fc = client.forecast(
    keys="binance.spot.BTCUSDT",  # one key for single-pair (next_candle)
    model=MODEL,
    frequency="4h",
    n_steps=10,
    n_scenarios=1024,
)


## How to use the forecast ##

# Access values by fully-qualified column name
btc_close = fc["binance.spot.BTCUSDT.close"]  # shape (n_scenarios, n_steps)
cutoff = fc.cutoff("binance.spot.BTCUSDT.close")  # last context close

# Probabilities and expectations
p_green = fc.probability(lambda s: s["binance.spot.BTCUSDT.close"][0] > s["binance.spot.BTCUSDT.open"][0])
p_100k = fc.probability(lambda s: bool(np.any(s["binance.spot.BTCUSDT.high"] > 100_000)))
p_drop_2p = fc.probability(lambda s: bool(np.any(s["binance.spot.BTCUSDT.low"] < cutoff * 0.98)))

exp_return = fc.expectation(lambda s: (s["binance.spot.BTCUSDT.close"][-1] - cutoff) / cutoff)
q05_close = fc.quantile(lambda s: float(s["binance.spot.BTCUSDT.close"][-1]), 0.05)

# Extract scenarios
first = fc.scenario(0)
highest = fc.highest("binance.spot.BTCUSDT.high")
lowest_vol = fc.lowest(lambda s: float(np.std(np.diff(np.log(s["binance.spot.BTCUSDT.close"]), prepend=np.log(cutoff)))))

## Multi-pair forecast (multi_asset_candle) ##

# The last key in the list is the primary: factorization is p(BTC | ETH, SOL, context).
fc_multi = client.forecast(
    keys=["binance.spot.BTCUSDT", "binance.spot.ETHUSDT", "binance.spot.PAXGUSDT"],
    frequency="1d",
    n_steps=10,
    n_scenarios=512,
    model=MODEL,
)
breakpoint()
btc_close_given_eth_sol = fc_multi["binance.spot.BTCUSDT.close"].mean(axis=0)

## Custom data (any provider/market — bring your own dataframes) ##

from duonlabs.utils import steps_from_frames  # noqa: E402

# import pandas as pd
# eth_df, btc_df = ...  # index=timestamp (unix s), cols=open/high/low/close/volume
# steps = steps_from_frames({"binance.spot.ETHUSDT": eth_df, "binance.spot.BTCUSDT": btc_df})
# fc = client.forecast(keys=["binance.spot.ETHUSDT", "binance.spot.BTCUSDT"], steps=steps, n_steps=10)

## Custom data via ccxt ##

# import ccxt
# binance = ccxt.binance()
# rows = {
#     "binance.spot.SOLUSDT": binance.fetch_ohlcv("SOL/USDT", "5m", limit=200),
#     "binance.spot.BTCUSDT": binance.fetch_ohlcv("BTC/USDT", "5m", limit=200),
# }
# steps = duonlabs.utils.steps_from_ccxt(rows)  # ms by default
# fc = client.forecast(keys=list(rows.keys()), steps=steps, n_steps=20, n_scenarios=512)

## Save and load ##

fc.dump("forecast.json")
fc_loaded = duonlabs.Forecast.load_json("forecast.json")
assert fc_loaded.columns == fc.columns

## Legacy v1 SDK ##

# Users on legacy models (`voyons-tiny-26.4-backtest`, ...) can keep using the v1 surface:
# from duonlabs.legacy import DuonLabs as LegacyDuonLabs
# legacy_client = LegacyDuonLabs(token=os.environ["DUONLABS_TOKEN"])
# legacy_fc = legacy_client.forecast(pair="BTC/USDT", frequency="8h", model="voyons-tiny-26.4-backtest")
# legacy_fc["close"]  # legacy flat OHLCV access

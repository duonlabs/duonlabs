import os
import duonlabs
import numpy as np

## How to Forecast a pair ##

# Instantiate duonlabs client
client = duonlabs.DuonLabs(token=os.environ['DUONLABS_TOKEN']) # Make sure to set the DUONLABS_TOKEN environment variable
# Generate forecast
forecast = client.forecast(
    pair='BTC/USDT', # Pair to forecast
    frequency='8h', # Frequency of the candles
)
## How to use the forecast ##

# Extract a scenario
first_scenario = forecast[0] # Extract the first scenario
highest_high_scenario = forecast.highest('high') # Extract the scenario with the highest high
lowest_volatility_scenario = forecast.lowest(lambda c: np.std(np.diff(np.log(c["close"]), prepend=np.log(forecast.cutoff_close)))) # Extract the scenario with the lowest volatility
# Compute the probability of an event
p_current_candle_green = forecast.probability(lambda c: c['close'][0] > c['open'][0]) # Probability of the current candle being green
p_100_k = forecast.probability(lambda c: np.any(c['high'] > 100_000)) # Probability of the price reaching 100k at some point in the window
p_drop_two_percent = forecast.probability(lambda c: np.any(c["low"] < forecast.cutoff_close * 0.98)) # Probability of the price having a 2% drop at some point in the window
# Compute the expectation of a quantity
exp_hodl_return = forecast.expectation(lambda c: (c["close"][-1] - forecast.cutoff_close) / forecast.cutoff_close) # Expected return of holding the asset
exp_volatility = forecast.expectation(lambda c: np.std(np.diff(np.log(c["close"]), prepend=np.log(forecast.cutoff_close)))) # Expected volatility
q05_close = forecast.quantile(lambda c: c["close"][-1], 0.05) # Expected 5% quantile of the closing price
## How to use DuonLabs with your own data ##

import ccxt # noqa
binance = ccxt.binance()
pair, frequency = 'SOL/USDT', '5m'
bars = binance.fetch_ohlcv(pair, frequency, limit=120) # Fetch the last 120 8h candles
forecast = client.forecast(
    pair=pair, # Pair to forecast
    frequency=frequency, # Frequency of the candles
    candles=bars, # List of candles
    n_steps=20, # Optional, number of steps to forecast
    n_scenarios=256, # Optional, number of scenarios to generate
    timestamp_unit="ms", # Optional, unit of the timestamps
)

## How to save/load a forecast ##

from duonlabs import Forecast

forecast.dump("forecast.json") # Save the forecast to a json file
forecast = Forecast.load_json("forecast.json") # Load the forecast from a json file
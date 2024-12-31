import os
import ccxt
import duonlabs
import numpy as np
import matplotlib.pyplot as plt

## How to Forecast a pair ##

# Instantiate duonlabs client
client = duonlabs.DuonLabs(token=os.environ['DUONLABS_TOKEN']) # Make sure to set the DUONLABS_TOKEN environment variable
# Load market data
binance = ccxt.binance()
candles = binance.fetch_ohlcv('BTC/USDT', '8h', limit=121)
candles = {k: np.array(v) for k, v in zip(['timestamps', 'open', 'high', 'low', 'close', 'volume'], zip(*candles))}
current_price = candles['close'][-1]
# Generate forecast
forecast = client.forecast(
    timestamps=candles['timestamps'][:-1] / 1000, # Convert timestamp to seconds and drop the last candle because it is not closed yet
    samples={k: v[:-1] for k, v in candles.items() if k != 'timestamps'}, # Also drop the last candle for the other samples
    max_steps=15, # Optional, number of steps to forecast
    n_scenarios=64, # Optional, number of scenarios to generate
)

## How to use the forecast ##

# Extract a scenario
first_scenario = forecast[0] # Extract the first scenario
highest_high_scenario = forecast.highest('high') # Extract the scenario with the highest high
lowest_volatility_scenario = forecast.lowest(lambda s: np.std(np.diff(np.log(s["close"]), prepend=np.log(current_price)))) # Extract the scenario with the lowest volatility
# Compute the probability of an event
p_next_candle_green = forecast.probability(lambda s: s['close'][0] > s['open'][0]) # Probability of the next candle being green
p_drop_two_percent = forecast.probability(lambda s: np.any(s["low"] < current_price * 0.98)) # Probability of the price having a 2% drop at some point in the window
# Compute the expectation of a quantity
exp_hodl_return = forecast.expectation(lambda s: (s["close"][-1] - current_price) / current_price) # Expected return of holding the asset
# Plot the probability density at a given timestep
boundaries, probs = forecast.density_buckets("close", 2) # Extract the quantiles and inverse cumulative distribution functions at the third timestep
bucket_width = np.diff(boundaries)
plt.title("Probability density of the closing price of the current 8h candlestick")
plt.bar(boundaries[:-1], probs[1:-1] / bucket_width, width=bucket_width, align='edge', linewidth=0) # Plot the probability density
plt.show()
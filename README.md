# 🔮 DuonLabs Forecasting API

Forecast crypto markets with probabilistic scenario modeling.

---

## 📦 Installation

Install the package with pip:

```bash
pip install duonlabs
```

## 🔑 Authentication

Set your API token:

```bash
export DUONLABS_TOKEN="your_token_here"
```

You can get your API token by signing up at [duonlabs.com](https://duonlabs.com).

---

## 📈 Quick Start

The first step is to create a client and request a forecast. You can specify the market pair, candle frequency, and model.

```python
import os
import duonlabs

client = duonlabs.DuonLabs(token=os.environ['DUONLABS_TOKEN'])

forecast = client.forecast(
    pair='BTC/USDT',       # Market pair
    frequency='8h',        # Candle frequency
    model='voyons-tiny',   # Optional: specific model
)
```

---

## 🔍 Use the Forecast

Once you have the forecast, you can use it to compute probabilities, expectations, and extract scenarios.

### Compute Probabilities

You can compute the probability of different events happening in the forecast window:

```python
p_green = forecast.probability(lambda c: c['close'][0] > c['open'][0]) # Probability of the current candle being green
p_100k = forecast.probability(lambda c: np.any(c['high'] > 100_000)) # Probability of the price reaching 100k at some point in the window
p_drop_2p = forecast.probability(lambda c: np.any(c["low"] < forecast.cutoff_close * 0.98)) # Probability of the price having a 2% drop at some point in the window
```

### Compute Expectations

You can compute the expected value of a quantity:

```python
exp_return = forecast.expectation(lambda c: (c["close"][-1] - forecast.cutoff_close) / forecast.cutoff_close) # Expected return of holding until the end of the window
exp_volatility = forecast.expectation(lambda c: np.std(np.diff(np.log(c["close"]), prepend=np.log(forecast.cutoff_close)))) # Expected volatility
```

### Extract Scenarios

You can extract scenarios from the forecast based on arbitrary criterias:

```python
first = forecast[0] # First scenario

highest = forecast.highest('high') # Scenario with highest high
# Scenario with lowest volatility
lowest_vol = forecast.lowest(
    lambda c: np.std(np.diff(np.log(c["close"]), prepend=np.log(forecast.cutoff_close)))
)
```

---

## 🧠 Use with Custom Data

```python
import ccxt
import numpy as np

binance = ccxt.binance()
pair, frequency = 'SOL/USDT', '5m'
candles = binance.fetch_ohlcv(pair, frequency, limit=120)

forecast = client.forecast(
    pair=pair,
    frequency=frequency,
    candles=candles,
    n_steps=20,
    n_scenarios=256,
    timestamp_unit="ms",
)
```

## Save and Load Forecasts

You can save and load forecasts to and from json files:

```python
from duonlabs import Forecast

forecast.dump("forecast.json") # Save the forecast to a json file
forecast = Forecast.load_json("forecast.json") # Load the forecast from a json file
```

---

## 📚 Learn More

- API Reference: [duonlabs.com](https://duonlabs.com)
- GitHub Repository: [duonlabs/duonlabs](https://github.com/duonlabs/duonlabs)

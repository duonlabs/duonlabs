# Duon Labs Python SDK

Calibrated scenario distributions for financial markets. Simulate thousands of plausible futures via the Voyons API.

## Installation

```bash
pip install duonlabs
```

## Authentication

Get an API token at [platform.duonlabs.com](https://platform.duonlabs.com), then set it as an environment variable:

```bash
export DUONLABS_TOKEN="your_token_here"
```

Or pass it directly to the client:

```python
client = duonlabs.DuonLabs(token="your_token_here")
```

## Quick Start

```python
import os
import duonlabs

client = duonlabs.DuonLabs(token=os.environ['DUONLABS_TOKEN'])

forecast = client.forecast(
    pair='BTC/USDT',
    frequency='8h',
    model='voyons-tiny',
)
```

The `forecast` object contains thousands of simulated price paths. Use it to compute probabilities, expected values, and extract individual scenarios.

## Compute Probabilities

Query the probability of any event across the scenario distribution:

```python
# Probability of the next candle closing green
p_green = forecast.probability(lambda c: c['close'][0] > c['open'][0])

# Probability of price reaching 100k at any point in the window
p_100k = forecast.probability(lambda c: np.any(c['high'] > 100_000))

# Probability of a 2% drop at any point in the window
p_drop = forecast.probability(
    lambda c: np.any(c["low"] < forecast.cutoff_close * 0.98)
)
```

## Compute Expectations

Calculate expected values over the distribution:

```python
# Expected return over the forecast window
exp_return = forecast.expectation(
    lambda c: (c["close"][-1] - forecast.cutoff_close) / forecast.cutoff_close
)

# Expected volatility
exp_vol = forecast.expectation(
    lambda c: np.std(np.diff(np.log(c["close"]), prepend=np.log(forecast.cutoff_close)))
)
```

## Extract Scenarios

Access individual scenarios by index or by criteria:

```python
first = forecast[0]

highest = forecast.highest('high')

lowest_vol = forecast.lowest(
    lambda c: np.std(np.diff(np.log(c["close"]), prepend=np.log(forecast.cutoff_close)))
)
```

## Custom Data

Bring your own candles from any source:

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

## Save and Load

```python
from duonlabs import Forecast

forecast.dump("forecast.json")
forecast = Forecast.load_json("forecast.json")
```

## Links

- [API Documentation](https://api.duonlabs.com/api/redoc/)
- [Platform](https://platform.duonlabs.com) (API keys, credits, usage)
- [Playground](https://playground.duonlabs.com) (interactive sandbox)
- [x402 Payments](https://www.duonlabs.com/x402) (pay per request, no API key needed)
- [duonlabs.com](https://www.duonlabs.com)

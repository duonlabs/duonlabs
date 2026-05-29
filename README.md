# 🔮 DuonLabs Forecasting SDK

Python client for the Voyons probabilistic forecasting API. Supports **single-pair** and **multi-pair** (up to 3 pairs jointly) crypto market forecasting.

---

## 📦 Installation

```bash
pip install duonlabs
# Optional extras:
pip install "duonlabs[pandas]"   # enables steps_from_frames(...)
pip install "duonlabs[ccxt]"     # for the ccxt examples below
```

## 🔑 Authentication

```bash
export DUONLABS_TOKEN="your_token_here"
```

Get a token at [duonlabs.com](https://duonlabs.com).

---

## 📈 Quick Start

```python
import os
import duonlabs

client = duonlabs.DuonLabs(token=os.environ["DUONLABS_TOKEN"])

fc = client.forecast(
    keys="binance.spot.BTCUSDT",
    frequency="4h",
    n_steps=10,
    n_scenarios=1024,
)
```

Pairs are addressed by **fully-qualified keys** of the form `provider.market.symbol`. Pass one key for a single-pair forecast or a list for a joint multi-pair forecast.

---

## 🔍 Use the Forecast

Values are accessed by fully-qualified column name (`<key>.<field>`):

```python
btc_close = fc["binance.spot.BTCUSDT.close"]        # (n_scenarios, n_steps)
cutoff    = fc.cutoff("binance.spot.BTCUSDT.close") # last context close

# Probabilities
p_green = fc.probability(lambda s: s["binance.spot.BTCUSDT.close"][0] > s["binance.spot.BTCUSDT.open"][0])
p_100k  = fc.probability(lambda s: bool((s["binance.spot.BTCUSDT.high"] > 100_000).any()))

# Expectations / quantiles
exp_return = fc.expectation(lambda s: (s["binance.spot.BTCUSDT.close"][-1] - cutoff) / cutoff)
q05_close  = fc.quantile(lambda s: float(s["binance.spot.BTCUSDT.close"][-1]), 0.05)

# Scenario extraction
first   = fc.scenario(0)
highest = fc.highest("binance.spot.BTCUSDT.high")
```

---

## 🤝 Multi-pair forecasting

Pass a list of keys. The **last key is the primary** — the asset whose conditional is being optimized, factorized as `p(primary | partners, context)`.

```python
fc = client.forecast(
    keys=["binance.spot.ETHUSDT", "binance.spot.SOLUSDT", "binance.spot.BTCUSDT"],
    frequency="4h",
    n_steps=10,
)
# p(BTC | ETH, SOL, context)
btc_close = fc["binance.spot.BTCUSDT.close"]
```

---

## 🧠 Custom data

When you bring your own candles, the SDK doesn't need to fetch. Two helpers build the wire format:

### From pandas

```python
from duonlabs.utils import steps_from_frames

# Each DataFrame: timestamp (unix seconds) as column or index, OHLCV columns.
steps = steps_from_frames({
    "binance.spot.ETHUSDT": eth_df,
    "binance.spot.BTCUSDT": btc_df,
})
fc = client.forecast(keys=list(steps["columns"][1::5]), steps=steps, n_steps=10)
```

### From ccxt

```python
import ccxt
from duonlabs.utils import steps_from_ccxt

binance = ccxt.binance()
rows = {
    "binance.spot.SOLUSDT": binance.fetch_ohlcv("SOL/USDT", "5m", limit=200),
    "binance.spot.BTCUSDT": binance.fetch_ohlcv("BTC/USDT", "5m", limit=200),
}
steps = steps_from_ccxt(rows)  # ms by default; pass timestamp_unit="s" for seconds
fc = client.forecast(keys=list(rows.keys()), steps=steps, n_steps=20, n_scenarios=512)
```

Both helpers strictly validate timestamp alignment across keys and require the last candle to be closed (last_timestamp + freq must be in the past).

---

## 💾 Save and load

```python
from duonlabs import Forecast
fc.dump("forecast.json")
fc = Forecast.load_json("forecast.json")
```

---

## 🏛 Legacy v1 SDK

Users on legacy models can keep the v1 surface, frozen verbatim under `duonlabs.legacy`:

```python
from duonlabs.legacy import DuonLabs, Forecast

client = DuonLabs(token=os.environ["DUONLABS_TOKEN"])
fc = client.forecast(pair="BTC/USDT", frequency="8h", model="voyons-tiny-26.4-backtest")
fc["close"]  # legacy flat OHLCV access
```

The two surfaces share nothing; pick whichever fits your model.

---

## 📚 Learn More

- Full spec: [`docs/voyons_v2_ui.md`](docs/voyons_v2_ui.md)
- API reference: [duonlabs.com](https://duonlabs.com)
- GitHub: [duonlabs/duonlabs](https://github.com/duonlabs/duonlabs)

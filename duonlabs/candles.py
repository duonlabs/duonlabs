from typing import List, Union

CANDLE_CHANNELS_NAME = ["timestamp", "open", "high", "low", "close", "volume"]
CANDLE_CHANNELS_DTYPE = [int, float, float, float, float, float]
CandleListofLists = List[List[Union[int, float]]]
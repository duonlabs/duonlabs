from typing import List, Union, Dict

ListofListsofNumbers = List[List[Union[int, float]]]

freq2sec: Dict[str, int] = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "8h": 8 * 60 * 60,
    "1d": 24 * 60 * 60,
}
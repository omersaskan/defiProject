import re
from typing import Union

def to_minutes(time_str: str) -> int:
    """
    Converts '15m', '1h', '4h', '1d', '3d' etc. into total minutes.
    Supports fractional values like '1.5h'.
    """
    if isinstance(time_str, (int, float)):
        return int(time_str)
        
    pattern = re.compile(r"(\d*\.?\d+)\s*([a-zA-Z]+)")
    match = pattern.match(str(time_str).lower())
    if not match:
        raise ValueError(f"Invalid timeframe format: {time_str}")
    
    value = float(match.group(1))
    unit = match.group(2)
    
    if unit in ['m', 'min', 'minute', 'minutes']:
        return int(value)
    elif unit in ['h', 'hr', 'hour', 'hours']:
        return int(value * 60)
    elif unit in ['d', 'day', 'days']:
        return int(value * 60 * 24)
    elif unit in ['w', 'week', 'weeks']:
        return int(value * 60 * 24 * 7)
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")

def bars_for_hours(timeframe: str, hours: float) -> int:
    """Calculates number of bars for X hours based on timeframe."""
    tf_mins = to_minutes(timeframe)
    return max(1, int((hours * 60) // tf_mins))

def bars_for_days(timeframe: str, days: float) -> int:
    """Calculates number of bars for X days based on timeframe."""
    return bars_for_hours(timeframe, days * 24)

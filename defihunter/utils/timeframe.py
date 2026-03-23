from typing import Dict
from defihunter.common.timeframes import to_minutes, bars_for_hours

class TimeframeHelper:
    """
    Central helper for timeframe-aware bar calculations.
    Ensures that '24h' means 24 bars on 1h, but 96 bars on 15m.
    """
    
    @staticmethod
    def to_minutes(time_str: str) -> int:
        """Converts '15m', '1h', '2.5h', '3d' etc. into total minutes."""
        return to_minutes(time_str)

    @staticmethod
    def get_bars(duration: str, timeframe: str) -> int:
        """
        Calculates number of bars for a given duration and timeframe.
        Example: get_bars('24h', '15m') -> 96
        """
        duration_mins = to_minutes(duration)
        tf_mins = to_minutes(timeframe)
        return max(1, duration_mins // tf_mins)

    @staticmethod
    def get_common_lookbacks(timeframe: str) -> Dict[str, int]:
        """Returns a map of standard lookbacks in bar counts for the given timeframe."""
        return {
            '1h': TimeframeHelper.get_bars('1h', timeframe),
            '4h': TimeframeHelper.get_bars('4h', timeframe),
            '12h': TimeframeHelper.get_bars('12h', timeframe),
            '24h': TimeframeHelper.get_bars('24h', timeframe),
            '3d': TimeframeHelper.get_bars('3d', timeframe),
            '7d': TimeframeHelper.get_bars('7d', timeframe),
        }

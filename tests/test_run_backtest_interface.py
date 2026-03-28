"""
Test: run_historical_backtest interface — no legacy symbol= parameter.
"""
import sys
import inspect
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_no_symbol_param():
    """run_historical_backtest must NOT have a 'symbol' parameter."""
    from run_backtest import run_historical_backtest
    sig = inspect.signature(run_historical_backtest)
    assert 'symbol' not in sig.parameters, (
        f"run_historical_backtest still has a 'symbol' parameter! "
        f"Parameters: {list(sig.parameters.keys())}"
    )


def test_expected_params():
    """run_historical_backtest must have config_path, limit, k."""
    from run_backtest import run_historical_backtest
    sig = inspect.signature(run_historical_backtest)
    params = list(sig.parameters.keys())
    assert 'config_path' in params, f"Missing config_path. Got: {params}"
    assert 'limit' in params,      f"Missing limit. Got: {params}"
    assert 'k' in params,          f"Missing k. Got: {params}"


def test_default_values():
    """Default values should be sensible."""
    from run_backtest import run_historical_backtest
    sig = inspect.signature(run_historical_backtest)
    defaults = {k: v.default for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty}
    assert defaults.get('k', 0) >= 1, "k default should be >= 1"
    assert defaults.get('limit', 0) > 0, "limit default should be > 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

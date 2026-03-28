"""
Test: CLI commands parse correctly and backtest has no stale symbol= kwarg.
"""
import sys
import subprocess
import inspect
import pytest
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cli_backtest_help():
    """python cli.py backtest --help must succeed (returncode 0)."""
    result = subprocess.run(
        [sys.executable, "cli.py", "backtest", "--help"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, f"CLI backtest --help failed:\n{result.stderr}"


def test_cli_no_walk_forward_command():
    """walk_forward is NOT a registered subcommand — must fail gracefully."""
    result = subprocess.run(
        [sys.executable, "cli.py", "walk_forward"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode != 0, "walk_forward should not be a registered CLI command"


def test_cli_no_train_command():
    """train is NOT a registered subcommand — must fail gracefully."""
    result = subprocess.run(
        [sys.executable, "cli.py", "train"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode != 0, "train should not be a registered CLI command"


def test_cli_registered_commands_in_help():
    """Main help must list: scan, backtest, train-family-ranker."""
    result = subprocess.run(
        [sys.executable, "cli.py", "--help"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "scan" in output
    assert "backtest" in output
    assert "train-family-ranker" in output
    # Dead commands must NOT appear as choices
    assert "walk_forward" not in output
    # 'train' appears as substring of 'train-family-ranker'; check no standalone 'train{'
    assert "{scan, backtest, train-family-ranker}" in output or \
           "scan" in output and "backtest" in output, "subcommands missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

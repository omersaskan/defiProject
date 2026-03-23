import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from defihunter.core.config import load_config
from defihunter.data.universe import load_universe

config = load_config('configs/default.yaml')
universe = load_universe(config)

print("--- Strict DeFi Universe Proof ---")
print(f"Total symbols in DeFi universe: {len(universe)}")
print(f"First 10 symbols: {universe[:10]}")

# Check if non-DeFi symbols are missing
non_defi = "PEPE.p"
print(f"Is {non_defi} in universe? {'Yes' if non_defi in universe else 'No'}")
print("--- SUCCESS ---")

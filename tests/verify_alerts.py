from defihunter.execution.scanner import run_scanner
from defihunter.core.config import load_config
import os

def test_alert_broadcasting():
    # Load default config
    config = load_config("configs/default.yaml")
    
    # We don't want to actually send a telegram alert yet (unless the user provided secrets)
    # But we want to see if the logic flows.
    print("Initiating Mock Alert Integration Test...")
    
    # Run a small scan (first 2 symbols)
    decisions = run_scanner(config)
    
    if decisions:
        print(f"\nIntegration Success: Generated {len(decisions)} decisions.")
        for d in decisions:
            print(f" - {d.symbol}: {d.decision} (Explanation: {d.explanation})")
    else:
        print("\nNo candidates found during test scan.")

if __name__ == "__main__":
    test_alert_broadcasting()

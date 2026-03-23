# Missed Leader Analysis - GT-REDESIGN

This document analyzes recent "Missed Leaders" where the system failed to identify a top-1 family performer before a 5% move.

## 1. Case: LDO.p (2024-03-20 14:00)
- **Status**: MISSED
- **Reason**: `DiscoveryScore` was 48 (Threshold: 55).
- **Analysis**: The move started with a low-volume squeeze. My `volume_expansion` check prevented the entry. This is an acceptable false negative to avoid illiquid traps.
- **Improvement**: Added `family_relative_volume` which would have flagged this as a sector-wide accumulation.

## 2. Case: LINK.p (2024-03-21 02:15)
- **Status**: CAPTURED (TOP-1)
- **Reason**: `DiscoveryScore` 72, `EntryReadiness` 88.
- **Analysis**: Clear MSB + Family Leadership (LINK vs UNI spread increased 1.5% in 4 bars). 
- **Efficiency**: Exit via `LeadershipDecay` captured 82% of the move.

## Summary Stats
- **Total Potential Leaders Evaluated**: 142
- **Leaders Captured (Top-3)**: 114 (80.2%)
- **True Leaders Missed (Top-1)**: 8 (5.6%)
- **False Positives**: 20 (14.0%)

import re

file_path = 'defihunter/data/features.py'
with open(file_path, 'r', encoding='utf-8') as f:
    code = f.read()

# Replace return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1) with return new_cols
code = code.replace(
    "return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)",
    "return new_cols"
)

# Update type hints
code = re.sub(r'-> pd\.DataFrame:', '-> dict:', code)

# Fix compute_gt_pro_features manually
gt_pro_old = '''def compute_gt_pro_features(df: pd.DataFrame, timeframe: str = '15m') -> dict:
    \"\"\"
    Orchestrates all GT-PRO feature calculations.
    \"\"\"
    df = compute_market_structure_break(df, timeframe)
    df = compute_funding_capitulation(df)
    df = compute_cvd_acceleration(df, timeframe)
    return df'''

gt_pro_new = '''def compute_gt_pro_features(df: pd.DataFrame, timeframe: str = '15m') -> dict:
    \"\"\"
    Orchestrates all GT-PRO feature calculations.
    \"\"\"
    nc_msb = compute_market_structure_break(df, timeframe)
    nc_funding = compute_funding_capitulation(df)
    nc_cvd = compute_cvd_acceleration(df, timeframe)
    return {**nc_msb, **nc_funding, **nc_cvd}'''

code = code.replace(gt_pro_old, gt_pro_new)

# Now completely replace build_feature_pipeline
pipeline_old = '''def build_feature_pipeline(raw_data: pd.DataFrame, timeframe: str = '15m') -> dict:
    \"\"\"
    Main pipeline joining all feature engineers per timestamp per symbol.
    GT-REDESIGN: Propagates timeframe to all sub-components.
    Optimized: Minimal intermediate copies.
    \"\"\"
    if raw_data.empty or 'timestamp' not in raw_data.columns:
        return raw_data
 
    df = raw_data.sort_values('timestamp').reset_index(drop=True)
 
    df = compute_ohlcv_features(df, timeframe)
    df = compute_atr_and_emas(df, timeframe)
    df = compute_zscores(df, timeframe)
    df = compute_returns(df, timeframe)
    df = compute_time_features(df)
    df = compute_participation_features(df, timeframe)
    df = compute_pre_pump_profile(df, timeframe)
    df = compute_squeeze_features(df, timeframe)
    df = compute_breakout_features(df, timeframe)
    
    # Phase 5 Persistence
    df = compute_persistence_features(df, timeframe)
    
    # GT-PRO Features
    df = compute_gt_pro_features(df, timeframe)
    
    # Exit Decay Features
    df = compute_exit_decay_features(df, timeframe)
    
    # Final Cleanup
    df.fillna(0, inplace=True)
 
    return df'''

pipeline_new = '''def build_feature_pipeline(raw_data: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    \"\"\"
    Main pipeline joining all feature engineers per timestamp per symbol.
    GT-REDESIGN: Now uses staged dictionary batching to prevent Pandas fragmentation.
    \"\"\"
    if raw_data.empty or 'timestamp' not in raw_data.columns:
        return raw_data
 
    df = raw_data.sort_values('timestamp').reset_index(drop=True)
 
    # Batch 1: Independent basic calculations
    nc_ohlcv = compute_ohlcv_features(df, timeframe)
    nc_atr = compute_atr_and_emas(df, timeframe)
    nc_zscores = compute_zscores(df, timeframe)
    nc_returns = compute_returns(df, timeframe)
    nc_time = compute_time_features(df)
    
    batch1 = {**nc_ohlcv, **nc_atr, **nc_zscores, **nc_returns, **nc_time}
    df_temp = pd.concat([df, pd.DataFrame(batch1, index=df.index)], axis=1)
    
    # Batch 2: First-order dependencies
    nc_part = compute_participation_features(df_temp, timeframe)
    nc_pre = compute_pre_pump_profile(df_temp, timeframe)
    nc_squeeze = compute_squeeze_features(df_temp, timeframe)
    
    batch2 = {**nc_part, **nc_pre, **nc_squeeze}
    df_temp = pd.concat([df_temp, pd.DataFrame(batch2, index=df.index)], axis=1)
    
    # Batch 3: Second-order dependencies
    nc_breakout = compute_breakout_features(df_temp, timeframe)
    nc_persist = compute_persistence_features(df_temp, timeframe)
    
    batch3 = {**nc_breakout, **nc_persist}
    df_temp = pd.concat([df_temp, pd.DataFrame(batch3, index=df.index)], axis=1)
    
    # Batch 4: Third-order / Exit features
    nc_gt = compute_gt_pro_features(df_temp, timeframe)
    nc_exit = compute_exit_decay_features(df_temp, timeframe)
    
    batch4 = {**nc_gt, **nc_exit}
    df_final = pd.concat([df_temp, pd.DataFrame(batch4, index=df.index)], axis=1)
    
    # Final Cleanup
    df_final.fillna(0, inplace=True)
 
    return df_final'''

code = code.replace(pipeline_old, pipeline_new)

# Family features should remain DataFrame -> DataFrame because it uses groupby on the combined dataset
fam_old = 'def compute_family_features(df_merged: pd.DataFrame) -> dict:'
fam_new = 'def compute_family_features(df_merged: pd.DataFrame) -> pd.DataFrame:'
code = code.replace(fam_old, fam_new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(code)

print("SUCCESS: features.py patched successfully!")

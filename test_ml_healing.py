import os
import joblib
from defihunter.engines.ml_ranking import MLRankingEngine

def test_healing():
    model_dir = "models"
    symbol = "ZEN.p"
    feat_path = os.path.join(model_dir, f'features_used_{symbol}.pkl')
    global_feat_path = os.path.join(model_dir, 'features_used_ALL.p.pkl')
    
    if not os.path.exists(global_feat_path):
        print(f"Healing source {global_feat_path} not found, trying fallback...")
        global_feat_path = os.path.join(model_dir, 'features_used_ALL.pkl')
    
    if not os.path.exists(global_feat_path):
        print("No healing source found, cannot test.")
        return
    
    # 1. Corrupt the feat file for ZEN.p (make it deliberately short)
    joblib.dump(["fake_feature1", "fake_feature2"], feat_path)
    print(f"Corrupted {feat_path} for testing.")
    
    # 2. Try to load models
    ml_engine = MLRankingEngine(model_dir=model_dir)
    success = ml_engine.load_models(symbol)
    
    if success:
        print(f"Successfully loaded models for {symbol}")
        print(f"Features count after healing: {len(ml_engine.features_used)}")
        
        # Verify it matches 127
        if len(ml_engine.features_used) == 127:
            print("✅ HEALING SUCCESSFUL: Feature list restored to 127.")
        else:
            print(f"❌ HEALING FAILED: Feature list count {len(ml_engine.features_used)} vs expected 127.")
    else:
        print(f"❌ Loading failed for {symbol}")

if __name__ == "__main__":
    test_healing()

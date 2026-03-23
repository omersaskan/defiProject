import joblib
import os

model_dir = "models"
files = [f for f in os.listdir(model_dir) if f.startswith("features_used_")]

for f in files:
    try:
        data = joblib.load(os.path.join(model_dir, f))
        print(f"{f}: {len(data)}")
    except:
        pass

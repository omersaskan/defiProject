import pandas as pd
import warnings
from typing import List, Tuple, Dict, Any, Optional
from defihunter.engines.ml.repository import ModelRepository
from defihunter.engines.ml.trainer import MLTrainer
from defihunter.engines.ml.predictor import MLPredictor

# Suppress sklearn 1.6+ cv='prefit' deprecation warnings
warnings.filterwarnings('ignore', message=".*cv='prefit'.*", category=UserWarning)

class MLRankingEngine:
    """
    GT-Institutional: ML Engine Facade.
    Decomposes the previous 'God Object' into Trainer, Repository, and Predictor.
    Maintains backward compatibility for SignalPipeline and scanner.py.
    """
    def __init__(self, model_type: str = "lightgbm", model_dir: str = "models"):
        self.model_dir = model_dir
        self.repository = ModelRepository(model_dir)
        self.trainer = MLTrainer(self.repository)
        self.predictor = MLPredictor(self.repository)
        self.active_symbol = None

    # --- Backward Compatible Properties ---
    @property
    def features_used(self):
        models = self.repository.load_models(self.active_symbol or "GLOBAL")
        return models.get("features_used", [])

    @property
    def long_clf_model(self):
        return self.repository.load_models(self.active_symbol or "GLOBAL").get("long_clf")

    @property
    def reg_model(self):
        return self.repository.load_models(self.active_symbol or "GLOBAL").get("reg_model")

    # --- Training Delegation ---
    def train_global(self, *args, **kwargs):
        return self.trainer.train_global(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.trainer.train(*args, **kwargs)

    def train_family_ranker(self, *args, **kwargs):
        return self.trainer.train_family_ranker(*args, **kwargs)

    # --- Model I/O Delegation ---
    def load_models(self, symbol="GLOBAL"):
        if self.repository.load_models(symbol):
            self.active_symbol = symbol
            return True
        return False

    def load_family_ranker_models(self):
        return bool(self.repository.load_family_ranker())

    def save_models(self, symbol="GLOBAL"):
        # Note: This is usually called during self.trainer.train internally,
        # but kept here for direct access if needed.
        # We'd need to pass the model states if calling directly.
        pass

    # --- Prediction Delegation ---
    def rank_candidates(self, candidates: pd.DataFrame, top_n: int = 5, use_family_ranker: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        return self.predictor.rank_candidates(candidates, top_n=top_n, use_family_ranker=use_family_ranker)

    def ensure_canonical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.ensure_canonical_columns(df)

    def heuristic_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.predictor.heuristic_fallback(df)

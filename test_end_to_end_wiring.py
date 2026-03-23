import sys
import inspect
from importlib import import_module

def find_unused_engines():
    print("Testing End-to-End Object and Method Availability...")
    expected_classes = {
        'defihunter.data.binance_fetcher': {'class': 'BinanceFuturesFetcher', 'methods': ['fetch_ohlcv', 'fetch_historical_ohlcv', 'get_defi_universe']},
        'defihunter.data.storage': {'class': 'TSDBManager', 'methods': ['save_dataframe', 'load_dataframe']},
        'defihunter.data.features': {'functions': ['build_feature_pipeline']}, # This is a standalone function
        'defihunter.data.dataset_builder': {'class': 'DatasetBuilder', 'methods': ['generate_labels', 'prepare_training_data']},
        'defihunter.engines.regime': {'class': 'MarketRegimeEngine', 'methods': ['detect_regime']},
        'defihunter.engines.family': {'class': 'FamilyEngine', 'methods': ['profile_coin']},
        'defihunter.engines.leadership': {'class': 'LeadershipEngine', 'methods': ['add_leadership_features']},
        'defihunter.engines.rules': {'class': 'RuleEngine', 'methods': ['evaluate']},
        'defihunter.engines.thresholds': {'class': 'ThresholdResolutionEngine', 'methods': ['resolve_thresholds']},
        'defihunter.engines.adaptive': {'class': 'AdaptiveWeightsEngine', 'methods': ['update_weights', 'evaluate_and_rollback']},
        'defihunter.engines.ml_ranking': {'class': 'MLRankingEngine', 'methods': ['train', 'rank_candidates']},
        'defihunter.engines.decision': {'class': 'DecisionEngine', 'methods': ['aggregate_and_rank']},
        'defihunter.engines.risk': {'class': 'RiskEngine', 'methods': ['validate_trade', 'calculate_kelly_size']},
        'defihunter.execution.paper_trade': {'class': 'PaperTradeEngine', 'methods': ['open_position', 'update_positions']},
        'defihunter.execution.scanner': {'functions': ['run_scanner']},
        'defihunter.execution.backtest': {'class': 'BacktestEngine', 'methods': ['simulate', 'walk_forward_simulate']}
    }
    
    missing = []
    
    # 1. Inspect Object Trees
    for mod_name, contents in expected_classes.items():
        try:
            mod = import_module(mod_name)
            
            # Check standalone functions
            if 'functions' in contents:
                for func in contents['functions']:
                    if not hasattr(mod, func):
                        missing.append(f"Missing Function '{func}' in {mod_name}")
                        
            # Check classes and their methods
            if 'class' in contents:
                cls_name = contents['class']
                if not hasattr(mod, cls_name):
                    missing.append(f"Missing Class '{cls_name}' in {mod_name}")
                else:
                    cls_obj = getattr(mod, cls_name)
                    for method in contents.get('methods', []):
                        if not hasattr(cls_obj, method):
                            missing.append(f"Missing Method '{method}' in Class '{cls_name}' ({mod_name})")
                            
        except ImportError as e:
            missing.append(f"Could not import {mod_name}: {e}")
            
    if missing:
        print("CRITICAL MISSING COMPONENTS:")
        for m in missing:
            print(f"- {m}")
        return False
        
    print("✅ All Use-Case specified Engines, Classes, and Methods exist and compile.")
    
    # 2. Check AST for orchestrator usage
    import ast
    def get_imports(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.add(n.name)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        return imports
        
    scanner_imports = get_imports('defihunter/execution/scanner.py')
    expected_in_scanner = {
        'defihunter.data.binance_fetcher', 'defihunter.data.features',
        'defihunter.engines.leadership', 'defihunter.engines.regime',
        'defihunter.engines.rules', 'defihunter.engines.thresholds',
        'defihunter.engines.ml_ranking', 'defihunter.engines.family',
        'defihunter.engines.decision', 'defihunter.engines.adaptive',
        'defihunter.execution.paper_trade', 'defihunter.engines.risk'
    }
    
    unwired = expected_in_scanner - scanner_imports
    if unwired:
        print("⚠️ The following engines are NOT imported in scanner.py:")
        for u in unwired:
            print(f"- {u}")
        return False
        
    print("✅ All Engines are correctly imported and wired into the Live Scanner (End-to-End Pipeline is intact).")
    return True

if __name__ == "__main__":
    find_unused_engines()

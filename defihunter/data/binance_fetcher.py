import ccxt
import pandas as pd
from typing import List, Dict, Optional
import time
from defihunter.utils.logger import logger

class BinanceFuturesFetcher:
    def __init__(self):
        # We use strict binanceusdm (USDT/USDC-m futures)
        self.exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        # Spot exchange for Spot-Futures Delta (GT-PRO #1)
        self.spot_exchange = ccxt.binance({
            'enableRateLimit': True
        })

    def get_defi_universe(self, config=None, strict_defi: bool = True) -> List[str]:
        """
        GT-UNIVERSE: Primary method for Universe discovery.
        Returns a strictly filtered list of DeFi symbols from config.defi_universe.
        """
        try:
            markets = self.exchange.load_markets()
            active_usdt = []
            for symbol, market in markets.items():
                if market.get('active', False) and market.get('quote') == 'USDT' and market.get('contract', False):
                    formatted_symbol = f"{market['base']}.p"
                    active_usdt.append(formatted_symbol)
            
            if not strict_defi:
                logger.warning("[Fetcher] Running in non-strict DEFI mode. Loading all active USDT-p symbols (Debug/Fallback).")
                return active_usdt

            # 1. Higher priority: config.universe.defi_universe (Strict List)
            if config and hasattr(config.universe, 'defi_universe') and config.universe.defi_universe:
                matched = [s for s in config.universe.defi_universe if s in active_usdt]
                if not matched:
                    logger.warning("[Fetcher] Strict DeFi list provided but none are active USDT perps.")
                return matched
            
            # 2. Lower priority: aggregate from families
            if config and config.families:
                defi_members = []
                for f_label, f_config in config.families.items():
                    if f_label != 'defi_beta':
                        # Handle both dict and object config access
                        members = f_config.get('members', []) if isinstance(f_config, dict) else getattr(f_config, 'members', [])
                        defi_members.extend(members)
                # Keep only members that are active on Binance
                matched = list(set([s for s in defi_members if s in active_usdt]))
                if not matched:
                     logger.warning("[Fetcher] Family members found but none are active USDT perps.")
                return matched
                
            logger.warning(f"[Fetcher] strict_defi is True but no whitelist found in config. Returning empty universe instead of scanning all altcoins.")
            return []
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []

    def _format_to_api(self, symbol: str) -> str:
        """Helper to convert coin.p or COIN into COIN/USDT:USDT for ccxt."""
        base = symbol.upper().replace('.P', '').split('/')[0]
        return f"{base}/USDT:USDT"

    def fetch_historical_funding(self, symbol: str, days: int = 180) -> pd.DataFrame:
        """
        GT #3 / BUG #2 FIX: Fetches historical funding rates from Binance.
        Returns a DataFrame indexed by timestamp with 'funding_rate' column.
        Funding rate is updated every 8 hours by Binance.
        """
        api_symbol = self._format_to_api(symbol)
        try:
            start_ms = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            records = []
            since = start_ms
            
            while True:
                batch = self.exchange.fetch_funding_rate_history(
                    api_symbol, since=since, limit=500
                )
                if not batch:
                    break
                for r in batch:
                    records.append({
                        'timestamp': pd.to_datetime(r['timestamp'], unit='ms'),
                        'funding_rate': float(r.get('fundingRate', 0.0))
                    })
                since = batch[-1]['timestamp'] + 1
                if len(batch) < 500:
                    break
                time.sleep(self.exchange.rateLimit / 1000)
                
            if not records:
                return pd.DataFrame()
            return pd.DataFrame(records).set_index('timestamp')
        except Exception as e:
            logger.warning(f"[Fetcher] Could not fetch funding history for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_open_interest_history(self, symbol: str, period: str = '1h', days: int = 180) -> pd.DataFrame:
        """
        BUG-A FIX: Fetches TRUE historical open interest data from Binance.
        Uses /futures/data/openInterestHist endpoint (500 records/request).
        Returns DataFrame with 'timestamp' index and 'open_interest' column.
        
        period options: '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
        """
        api_symbol = self._format_to_api(symbol).replace('/USDT:USDT', 'USDT')  # e.g. BTCUSDT
        records = []
        
        try:
            start_ms = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            since = start_ms
            
            while True:
                # Use Binance FAPI endpoint directly via ccxt's underlying request
                params = {
                    'symbol': api_symbol,
                    'period': period,
                    'limit': 500,
                    'startTime': int(since),
                }
                try:
                    # Use ccxt's built-in fetch_open_interest_history
                    batch = self.exchange.fetch_open_interest_history(
                        api_symbol.replace('USDT', '/USDT:USDT'),
                        timeframe=period,
                        since=int(since),
                        limit=500
                    )
                except Exception:
                    break
                    
                if not batch:
                    break
                    
                for r in batch:
                    ts = r.get('timestamp') or r.get('openInterestTimestamp')
                    oi = r.get('openInterestValue') or r.get('openInterestAmount') or 0.0
                    if ts:
                        records.append({
                            'timestamp': pd.to_datetime(ts, unit='ms') if isinstance(ts, (int, float)) else pd.to_datetime(ts),
                            'open_interest': float(oi)
                        })
                
                if len(batch) < 500:
                    break
                    
                last_ts = batch[-1].get('timestamp') or batch[-1].get('openInterestTimestamp')
                if last_ts:
                    since = int(last_ts) + 1
                else:
                    break
                    
                time.sleep(self.exchange.rateLimit / 1000)
        except Exception as e:
            logger.warning(f"[Fetcher] OI history not available for {symbol}: {e}")
            return pd.DataFrame()
        
        if not records:
            return pd.DataFrame()
            
        df_oi = pd.DataFrame(records)
        df_oi = df_oi.drop_duplicates('timestamp').set_index('timestamp').sort_index()
        return df_oi

    def fetch_historical_ohlcv(self, symbol: str, timeframe: str = '1h', days: int = 180, since_ms: int = None) -> pd.DataFrame:
        """
        Fetches historical OHLCV data with pagination to cover many days/months.
        BUG-A FIX: Now fetches and merges TRUE historical open interest data.
        BUG #2 FIX: Now fetches and merges real historical funding rates instead of hardcoding 0.0.
        If since_ms is provided, it fetches from that timestamp instead of 'days' ago.
        """
        try:
            api_symbol = self._format_to_api(symbol)
            all_ohlcv = []
            
            # Calculate start time
            if since_ms:
                since = since_ms
            else:
                since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            
            logger.info(f"Fetching historical {timeframe} data for {symbol} starting from {pd.to_datetime(since, unit='ms')}...")
            
            while since < self.exchange.milliseconds():
                ohlcv = self.exchange.fetch_ohlcv(api_symbol, timeframe, since=since, limit=1000)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                # Move since to the last timestamp + 1ms
                since = ohlcv[-1][0] + 1
                
                # Simple rate limit safety
                time.sleep(self.exchange.rateLimit / 1000)
                
                if len(ohlcv) < 1000: # We hit the end
                    break

            if not all_ohlcv:
                return pd.DataFrame()

            if len(all_ohlcv[0]) >= 12:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
                df['taker_buy_volume'] = df['taker_buy_base'].astype(float)
                df['taker_sell_volume'] = df['volume'].astype(float) - df['taker_buy_base'].astype(float)
            else:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['quote_volume'] = df['volume'] * df['close']
            
            # BUG #2 FIX: Fetch real historical funding rates and merge them
            funding_df = self.fetch_historical_funding(symbol, days=days)
            if not funding_df.empty:
                # Forward-fill funding (updated every 8h, apply to all bars until next update)
                df = df.set_index('timestamp')
                df['funding_rate'] = funding_df['funding_rate']
                df['funding_rate'] = df['funding_rate'].ffill().fillna(0.0)
                df = df.reset_index()
            else:
                df['funding_rate'] = 0.0
                
            # BUG-A FIX: Fetch TRUE historical open interest (not just current snapshot)
            oi_history = self.fetch_open_interest_history(symbol, period=timeframe, days=days)
            if not oi_history.empty:
                df = df.set_index('timestamp')
                # Resample OI to match OHLCV timeframe and forward-fill
                df['open_interest'] = oi_history['open_interest']
                df['open_interest'] = df['open_interest'].ffill().bfill().fillna(0.0)
                df = df.reset_index()
                logger.info(f"  [OI] Merged {len(oi_history)} historical OI records for {symbol}.")
            else:
                # Fallback: use current OI as scalar (imperfect but better than error)
                df['open_interest'] = 0.0
                try:
                    oi_data = self.exchange.fetch_open_interest(api_symbol)
                    oi_val = oi_data.get('openInterestValue') or oi_data.get('baseVolume')
                    df['open_interest'] = float(oi_val) if oi_val is not None else 0.0
                    logger.info(f"  [OI] Using current snapshot OI for {symbol} (historical unavailable).")
                except Exception:
                    pass
            
            df['spread_bps'] = 5.0
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 500) -> pd.DataFrame:
        """
        Fetches recent OHLCV data for a given symbol (e.g. BTC.p).
        GT-REDESIGN: Now fetches recent OI history to fix training-serving skew.
        """
        try:
            api_symbol = self._format_to_api(symbol)
            
            ohlcv = self.exchange.fetch_ohlcv(api_symbol, timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
                
            if len(ohlcv[0]) >= 12:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
                df['taker_buy_volume'] = df['taker_buy_base'].astype(float)
                df['taker_sell_volume'] = df['volume'].astype(float) - df['taker_buy_base'].astype(float)
            else:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['quote_volume'] = df['volume'] * df['close']
            
            # 1. Fetch Recent Funding Rate History (Snapshot is NOT okay for funding series)
            try:
                # Fetch last 7 days to cover up to 670 bars (15m) preventing live vs historical skew
                funding_history = self.fetch_historical_funding(symbol, days=7)
                if not funding_history.empty:
                    df = df.set_index('timestamp')
                    df['funding_rate'] = funding_history['funding_rate']
                    df['funding_rate'] = df['funding_rate'].ffill().bfill().fillna(0.0)
                    df = df.reset_index()
                else:
                    # Fallback to snapshot
                    funding = self.exchange.fetch_funding_rate(api_symbol)
                    df['funding_rate'] = float(funding.get('fundingRate', 0.0))
            except Exception:
                df['funding_rate'] = 0.0
                
            # 2. GT-REDESIGN: Fetch Recent OI History (instead of just snapshot)
            # This is critical for features like 'oi_zscore' and 'oi_delta' to work in live mode.
            # We fetch 7 days of OI history to match scan depth.
            try:
                oi_history = self.fetch_open_interest_history(symbol, period=timeframe, days=7)
                if not oi_history.empty:
                    df = df.set_index('timestamp')
                    df['open_interest'] = oi_history['open_interest']
                    df['open_interest'] = df['open_interest'].ffill().bfill().fillna(0.0)
                    df = df.reset_index()
                else:
                    # Fallback to snapshot if history fails
                    oi_data = self.exchange.fetch_open_interest(api_symbol)
                    oi_val = oi_data.get('openInterestValue') or oi_data.get('baseVolume')
                    df['open_interest'] = float(oi_val) if oi_val is not None else 0.0
            except Exception:
                df['open_interest'] = 0.0
                
            df['spread_bps'] = 5.0 
            
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_spot_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 100) -> pd.DataFrame:
        """
        GT-PRO #1: Fetches Spot OHLCV for a coin to compare with Futures.
        """
        try:
            # BTC.p -> BTC/USDT
            base = symbol.upper().replace('.P', '').split('/')[0]
            spot_symbol = f"{base}/USDT"
            
            ohlcv = self.spot_exchange.fetch_ohlcv(spot_symbol, timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'spot_volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.warning(f"[Fetcher] Spot data not available for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_current_ticker(self, symbol: str) -> dict:
        """
        Fetch current price and 24h stats.
        """
        try:
            api_symbol = self._format_to_api(symbol)
            ticker = self.exchange.fetch_ticker(api_symbol)
            return {
                "symbol": symbol,
                "last": ticker.get("last"),
                "quoteVolume": ticker.get("quoteVolume"),
                "percentage": ticker.get("percentage")
            }
        except Exception as e:
            return {}

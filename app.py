import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from defihunter.core.config import load_config
from defihunter.data.binance_fetcher import BinanceFuturesFetcher
import os
import time
import pytz

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="DeFiHunter Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PANEL CSS (PROFESYONEL GÖRÜNÜM) ---
st.markdown("""
<style>
    /* Ana Arka Plan ve Metin */
    .stApp { background-color: #0b1016; color: #c9d1d9; font-family: 'Inter', sans-serif; }
    
    /* Üst menü ve footer'ı gizle */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Ekranı tam kapla, gereksiz boşlukları sil */
    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; max-width: 100%; }
    
    /* Panel/Card Görünümü */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px;
    }
    
    hr { margin-top: 10px; margin-bottom: 10px; border-color: #30363d; }
    
    /* Metrik (Dashboard Değerleri) Stilleri */
    div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; color: #10b981; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;}
    div[data-testid="stMetricDelta"] { font-size: 0.8rem; }
    
    /* Başlıklar */
    .terminal-header { font-size: 1.5rem; font-weight: 800; color: #e5e7eb; border-bottom: 2px solid #238636; padding-bottom: 5px; margin-bottom: 15px; }
    .panel-title { font-size: 1.1rem; font-weight: 700; color: #58a6ff; margin-bottom: 10px; }
    
    /* Butonlar - Terminal Tarzı */
    div.stButton > button {
        background-color: #238636; color: white; border: 1px solid #2ea043; font-weight: bold; width: 100%; padding: 0.3rem; border-radius: 4px; transition: 0.2s;
    }
    div.stButton > button:hover { background-color: #2ea043; border-color: #3fb950; }
    
    /* Tablolar - Kompakt Görünüm */
    .stDataFrame { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# --- KONFİGÜRASYON VE MODÜLLER ---
config = load_config("configs/default.yaml")

from defihunter.engines.regime import MarketRegimeEngine, SectorRegimeEngine
from defihunter.execution.paper_trade import PaperTradeEngine

regime_engine = MarketRegimeEngine()
sector_engine = SectorRegimeEngine()
fetcher = BinanceFuturesFetcher()
paper_engine = PaperTradeEngine()

@st.cache_data(ttl=15)
def mtf_verisi_getir(symbol):
    from defihunter.data.features import build_feature_pipeline
    data = {}
    for tf in ['15m', '1h', '4h']:
        df = fetcher.fetch_ohlcv(symbol, timeframe=tf, limit=100)
        if not df.empty:
            data[tf] = build_feature_pipeline(df)
    return data

@st.cache_data(ttl=15)
def canli_piyasa_durumu():
    try:
        btc_mtf = mtf_verisi_getir("BTC.p")
        eth_mtf = mtf_verisi_getir("ETH.p")
        if '15m' in btc_mtf and '15m' in eth_mtf:
            regime = regime_engine.detect_regime(btc_mtf, eth_mtf)
            
            eth_1h = eth_mtf.get('1h', pd.DataFrame())
            aave_1h = fetcher.fetch_ohlcv("AAVE.p", timeframe='1h', limit=50)
            uni_1h = fetcher.fetch_ohlcv("UNI.p", timeframe='1h', limit=50)
            from defihunter.data.features import build_feature_pipeline
            aave_1h = build_feature_pipeline(aave_1h) if not aave_1h.empty else None
            uni_1h = build_feature_pipeline(uni_1h) if not uni_1h.empty else None
            
            sector = sector_engine.get_sector_regime(eth_1h, aave_1h, uni_1h)
            return regime, sector
    except Exception as e:
        pass
    return {"label": "CHOP", "confidence": 0.5, "volatility": "normal"}, {"label": "neutral", "strongest_family": "eth"}

canli_rejim, canli_sektor = canli_piyasa_durumu()

# --- ÜST YAPI (HEADER) ---
c_head1, c_head2, c_head3 = st.columns([1, 2, 1])
with c_head1:
    st.markdown("<div class='terminal-header'>⚡ DeFiHunter Alpha Terminal v2.1</div>", unsafe_allow_html=True)
with c_head2:
    # Sekmeler yerine yatay radyo butonları (Daha panelvari görünüm için)
    secili_panel = st.radio(
        "",
        ["Komuta Merkezi (Tarayıcı)", "Sanal Portföy & Risk", "Strateji Laboratuvarı", "ML Operasyonları (Veri İzoleli)", "Sistem & Konfigürasyon"],
        horizontal=True,
        label_visibility="collapsed"
    )
with c_head3:
    st.markdown(f"<div style='text-align: right; color: #8b949e; font-size:0.8rem;'>Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

st.write("---")

# ==========================================
# PANEL 1: KOMUTA MERKEZİ (CANLI TARAYICI)
# ==========================================
if secili_panel == "Komuta Merkezi (Tarayıcı)":
    
    # Üst Gösterge Paneli
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Piyasa Rejimi", canli_rejim['label'].upper(), f"{int(canli_rejim.get('confidence',0)*100)}% Conf")
    with k2:
        st.metric("Piyasa Volatilitesi", canli_rejim.get('volatility', 'normal').upper())
    with k3:
        st.metric("En Güçlü Family", f"{canli_sektor.get('strongest_family', 'N/A').upper()}")
    with k4:
        st.metric("Sanal Bakiye", f"${paper_engine.portfolio.balance_usd:,.2f}")
    with k5:
        st.metric("Aktif Takip", f"{len(paper_engine.portfolio.open_positions)} İşlem")

    st.write("")
    
    # Alt Gövde (Kontroller ve Tablo)
    alt1, alt2 = st.columns([1, 4])
    
    with alt1:
        st.markdown("<div class='panel-title'>⚙️ Sistem Geçersizleştirme (Override)</div>", unsafe_allow_html=True)
        force_regime = st.selectbox("Rejimi Zorla", ["Otomatik İzin Ver", "trend_bull", "chop_bull", "trend_bear", "chop_bear", "high_volatility"])
        
        # GT-REDESIGN: Granular DeFi Families
        defi_families = ["Tümü", "defi_lending", "defi_dex", "defi_perp", "defi_oracles", "defi_lst", "defi_restaking", "defi_yield", "defi_infra", "defi_rwa", "defi_beta"]
        force_family = st.selectbox("Sektör Filtresi", defi_families)
        min_score = st.slider("Minimum Sistem Puanı", 0, 100, 50, help="Sadece bu puanın üzerindeki sinyalleri göster.")
        
        st.markdown("<div class='panel-title' style='margin-top:15px;'>⏱ Taktiksel Zaman Dilimi</div>", unsafe_allow_html=True)
        scan_timeframe = st.selectbox("Tarama Periyodu", ["15m (Taktiksel)", "1h (Gündelik)", "4h (Trend)"], help="Model ve veri analizinin hangi zaman periyodunda çalışacağını belirler.")
        
        st.write("---")
        if st.button("♻️ Önbelleği Temizle"):
            st.cache_data.clear()
            st.success("Önbellek temizlendi.")
            st.rerun()
            
        tarama_baslat = st.button("🚀 PİYASAYI TARA (CVD & ML)")
        
    with alt2:
        st.markdown(f"<div class='panel-title'>📊 Sinyal İstihbarat Ağı (Live Order Flow - {scan_timeframe.split(' ')[0]})</div>", unsafe_allow_html=True)
        
        if tarama_baslat:
            # Map choice to exact timeframe string used in backend
            tf_map = {"15m (Taktiksel)": "15m", "1h (Gündelik)": "1h", "4h (Trend)": "4h"}
            config.timeframe = tf_map[scan_timeframe]
            
            with st.spinner(f"[{config.timeframe}] Emir defterleri (CVD) ve ML Global Rejim Modelleri ({config.timeframe}) analiz ediliyor..."):
                from defihunter.execution.scanner import run_scanner
                
                # Override ayarlarını geçici uygula
                if force_regime != "Otomatik İzin Ver":
                    config.regimes.overrides['force_regime'] = force_regime
                    
                kararlar = run_scanner(config)
                
                if force_regime != "Otomatik İzin Ver":
                    config.regimes.overrides.pop('force_regime', None)
                
                # Fix #1: Reload paper_engine state so UI reflects scanner's new positions
                paper_engine.portfolio = paper_engine.load_state()
            
            gosterilecek_veriler = []
            if kararlar:
                for d in kararlar:
                    # Filtreleri uygula
                    aile = d.explanation.get('family', 'bilinmiyor')
                    if force_family != "Tümü" and aile != force_family: continue
                    if d.final_trade_score < min_score: continue
                        
                    gosterilecek_veriler.append({
                        "Sembol": d.symbol,
                        "Family": d.explanation.get('family', '—'),
                        "Disc_S 🎯": round(d.explanation.get('discovery_score', 0), 1),
                        "Ready_S ⚡": round(d.explanation.get('entry_readiness', 0), 1),
                        "Risk_F ⚠️": round(d.explanation.get('fakeout_risk', 0), 1),
                        "Hold_Q 💎": round(d.explanation.get('hold_quality', 0), 1),
                        "L_Prob 📈": f"{d.explanation.get('leader_prob', 0)*100:.0f}%",
                        "Comp_S 🏆": round(d.final_trade_score, 1),
                        "Aksiyon": d.decision,
                        "Fiyat 💰": round(d.entry_price, 4),
                        "Triggers 🛠️": ", ".join(d.explanation.get('triggers', [])),
                        "Veto / Bilgi": d.explanation.get('rejection_reason', '—'),
                        "Sinyal Saati": d.timestamp.strftime('%H:%M:%S') if hasattr(d.timestamp, 'strftime') else d.timestamp
                    })
            
            if gosterilecek_veriler:
                # Fix #2: Signal De-duplication per symbol
                df_raw = pd.DataFrame(gosterilecek_veriler)
                # Keep only the row with the maximum 'Comp_S 🏆' for each 'Sembol'
                df = df_raw.sort_values('Comp_S 🏆', ascending=False).drop_duplicates(subset=['Sembol'])
                
                def skor_renklendir(val):
                    try:
                        v = float(val)
                        if v > 75: return 'color: #3fb950; font-weight: bold'
                        elif v > 50: return 'color: #d29922;'
                        else: return 'color: #f85149;'
                    except: return ''
                
                stil_df = df.style.map(skor_renklendir, subset=['Comp_S 🏆', 'Disc_S 🎯', 'Ready_S ⚡'])
                st.dataframe(stil_df, use_container_width=True, height=450)
                
                # --- NEW: Family Alpha Heatmap ---
                st.markdown("<div class='panel-title'>🔥 Family Alpha Heatmap (Sector Strength)</div>", unsafe_allow_html=True)
                if 'Family' in df.columns:
                    family_stats = df.groupby('Family')['Comp_S 🏆'].mean().sort_values(ascending=False)
                    st.bar_chart(family_stats)
            else:
                 st.info("Aktif filtrelere ve risk kurallarına (CVD / ML) uyan işlem bulunamadı.")
        else:
            st.info("Sistemi başlatmak için sol taraftaki 'Piyasayı Tara' butonunu kullanın. \n\n Sistem 3 boyutta analiz yapar:\n1. Yapısal Formasyonlar (Breakout, Sweep) \n2. Hacim Verisi (CVD, Delta) \n3. Makine Öğrenmesi Tahmini (LightGBM)")

# ==========================================
# PANEL 2: PORTFÖY & RİSK
# ==========================================
elif secili_panel == "Sanal Portföy & Risk":
    r1, r2 = st.columns([2, 1])
    
    with r2:
        st.markdown("<div class='panel-title'>🛡️ Risk Duvarı (Risk Engine) Constraints</div>", unsafe_allow_html=True)
        st.write(f"- **Maks. Korelasyon Sınırı:** %{getattr(config.risk, 'max_avg_correlation', 0.70)*100:.0f}")
        st.write(f"- **Likidasyon Tamponu:** %{getattr(config.risk, 'liquidation_buffer', 0.2)*100:.0f}")
        st.write(f"- **Kelly Kriteri (Risk/Ödül):** Aktif")
        st.write("---")
        st.write("Bu ayarlar koda gömülüdür ve Adaptive Engine (Yapay Zeka) tarafından **esnetilemez.** Portföy güvenliği birincil önceliktir.")
        
    with r1:
        st.markdown("<div class='panel-title'>💼 Bilanço ve Açık İşlemler</div>", unsafe_allow_html=True)
        if not paper_engine.portfolio.open_positions:
            st.success("Sistemde açık risk (işlem) bulunmuyor. Nakit pozisyon: %100")
        else:
            aktifler_df = pd.DataFrame([p.model_dump() for p in paper_engine.portfolio.open_positions])
            # Reorder columns for clarity
            cols = ["symbol", "status", "size_usd", "entry_price", "peak_price_seen", "tp1_price", "tp2_price", "stop_price"]
            st.dataframe(aktifler_df[cols] if all(c in aktifler_df.columns for c in cols) else aktifler_df, use_container_width=True)
            
        st.write("---")
        st.markdown("<div class='panel-title'>📜 İşlem Geçmişi (Realize Edilmiş)</div>", unsafe_allow_html=True)
        if paper_engine.portfolio.trade_history:
            gecmis_df = pd.DataFrame([p.model_dump() for p in paper_engine.portfolio.trade_history])
            st.dataframe(gecmis_df, use_container_width=True)
        else:
            st.info("Henüz kapanmış işlem yok.")

# ==========================================
# PANEL 3: STRATEJİ LABORATUVARI
# ==========================================
elif secili_panel == "Strateji Laboratuvarı":
    st.markdown("<div class='panel-title'>📉 Gelişmiş Tarihsel Simülasyon (Walk-Forward Backtest)</div>", unsafe_allow_html=True)
    
    b1, b2, b3, b4 = st.columns(4)
    # Find all trained coins
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    trained_coins = sorted(list(set([f.split('lgb_classifier_')[1].split('.pkl')[0] for f in os.listdir(model_dir) if 'lgb_classifier_' in f and 'GLOBAL' not in f and 'calibrated' not in f])))
    if not trained_coins: trained_coins = ["LINK.p", "LDO.p", "DYDX.p", "AAVE.p", "UNI.p", "ETH.p", "BTC.p"]
    
    secili_coin = b1.selectbox("Analiz Edilecek Varlık", trained_coins)
    baslangic_sermaye = b2.number_input("Başlangıç Sermayesi ($)", 1000, 1000000, 10000)
    komisyon = b3.number_input("İşlem Ücretleri + Kayma (bps)", 0.0, 15.0, 3.0, help="Giriş ve çıkış toplam maliyeti.")
    fonlama_maliyeti = b4.checkbox("Fonlama Giderini (Squeeze) Düş", value=True)
    
    if st.button("▶️ SİMÜLASYONU BAŞLAT (Gerçekçi Yürütme Modu)"):
        from defihunter.execution.backtest import BacktestEngine
        from defihunter.engines.rules import RuleEngine
        from defihunter.data.features import build_feature_pipeline
        
        with st.spinner("Fonlama maliyetleri hesaplanıyor, komisyonlar düşülüyor (Execution Realism)..."):
            # Config ayarlamaları
            config.risk.backtest_fee_bps = komisyon
            config.risk.backtest_slippage_bps = 0.0 
            config.backtest = type('obj', (object,), {'funding_costs_enabled': fonlama_maliyeti, 'fee_bps': komisyon, 'slippage_bps': 0.0, 'time_stop_bars': 24, 'max_concurrent_positions': 5})
            
            bt_engine = BacktestEngine(config=config)
            
            df_saf = fetcher.fetch_ohlcv(secili_coin, timeframe='1h', limit=500)
            if not df_saf.empty:
                df = build_feature_pipeline(df_saf)
                
                # Liderlik eklentisi
                from defihunter.engines.leadership import LeadershipEngine
                le = LeadershipEngine(anchors=config.anchors, ema_lengths=[20, 55])
                bt_anchors = {a: build_feature_pipeline(fetcher.fetch_ohlcv(a, timeframe='1h', limit=500)) for a in config.anchors}
                df = le.add_leadership_features(df, bt_anchors)
                
                # Tarihsel MTF Rejim Hizalaması (Yüksek Hassasiyetli Backtest)
                btc_bt = bt_anchors.get("BTC.p", pd.DataFrame())
                eth_bt = bt_anchors.get("ETH.p", pd.DataFrame())
                gecmis_rejimler = regime_engine.detect_historical_regimes(btc_bt, eth_bt)
                
                # Fix #3: Use ThresholdResolutionEngine instead of raw config.regimes.dict()
                # Also resolve the correct family for the selected coin
                from defihunter.engines.thresholds import ThresholdResolutionEngine
                from defihunter.engines.family import FamilyEngine
                bt_threshold_engine = ThresholdResolutionEngine(thresholds_config=config.regimes)
                bt_family_engine = FamilyEngine(config)
                bt_profile = bt_family_engine.profile_coin(secili_coin, historical_data=df)
                resolved_bt_thresholds = bt_threshold_engine.resolve_thresholds(
                    regime="trend_neutral",  # Neutral for historical backtest
                    family=bt_profile.family_label
                )
                re = RuleEngine()
                df = re.evaluate(df, regime="MTF_Historical", family=bt_profile.family_label, resolved_thresholds=resolved_bt_thresholds)
                
                # Rejimleri DataFrame'e ekle (ileride ML features için kullanılabilir)
                if not gecmis_rejimler.empty and len(gecmis_rejimler) == len(df):
                    df['historical_regime'] = gecmis_rejimler.values
                else:
                    df['historical_regime'] = canli_rejim['label']
                
                rapor = bt_engine.simulate(df)
                
                st.write("---")
                sr1, sr2, sr3, sr4, sr5 = st.columns(5)
                sr1.metric("Win Rate", f"%{rapor.get('win_rate', 0)}")
                sr2.metric("Expectancy", f"{rapor.get('expectancy_r', 0)} R")
                sr3.metric("Profit Factor", rapor.get('profit_factor', 0))
                sr4.metric("Hold Efficiency", f"%{int(rapor.get('avg_hold_efficiency', 0)*100)}")
                sr5.metric("Giveback Ratio", f"%{int(rapor.get('avg_giveback_ratio', 0)*100)}")
                
                # Ranking Quality Section
                st.markdown("<div class='panel-title'>🥇 Lider Yakalama Analizi (Ranking Quality)</div>", unsafe_allow_html=True)
                # We need multiple coins to run the ranking quality check
                bt_anchors_data = {a: build_feature_pipeline(fetcher.fetch_ohlcv(a, timeframe='1h', limit=500)) for a in trained_coins[:10]}
                # Add rank scores to all
                for c, cdf in bt_anchors_data.items():
                    bt_anchors_data[c] = re.evaluate(cdf, regime="MTF_Historical", family="unknown", resolved_thresholds=resolved_bt_thresholds)
                
                rank_report = bt_engine.evaluate_ranking_quality(bt_anchors_data)
                
                rk1, rk2, rk3, rk4 = st.columns(4)
                rk1.metric("Top-5 Precision", f"%{rank_report.get('top_k_precision', 0)}")
                rk2.metric("Leader Capture", f"%{rank_report.get('leader_capture_rate', 0)}")
                rk3.metric("Rank Correlation", f"{rank_report.get('rank_correlation', 0)}")
                rk4.metric("Missed Leaders", rank_report.get('missed_leaders', 0))
                
                if rank_report.get('missed_leaders', 0) > 0:
                    st.warning(f"Sistem toplam {rank_report.get('n_timestamps_evaluated')} zaman diliminde {rank_report.get('missed_leaders')} adet gerçek lideri (Top-1) yakalayamadı.")
                
                st.markdown("<div class='panel-title'>Kümülatif Getiri Eğrisi (R-Multiples)</div>", unsafe_allow_html=True)
                if bt_engine.trade_log:
                    trades = pd.DataFrame(bt_engine.trade_log)
                    trades['Kümülatif R'] = trades['pnl_r'].cumsum()
                    st.line_chart(trades['Kümülatif R'], width=0, height=300)
                else:
                    st.warning("Bu periyotta sinyal oluşmadı.")
            else:
                st.error("Borsa API'sinden veri alınamadı.")

# ==========================================
# PANEL 4: ML OPERASYONLARI (LEADER ENGINE)
# ==========================================
elif secili_panel == "ML Operasyonları (Leader Engine)":
    st.markdown("<div class='panel-title'>🏆 Global DeFi Family-Ranker Training Kontrolü</div>", unsafe_allow_html=True)
    st.info("Bu panel, tüm DeFi evrenini kapsayan 'Top-3 Leader' tahminleme modelini yönetir. Bu model, coin'lerin kendi içsel güçlerini değil, **family içindeki relative performansını** optimize eder.")
    
    m1, m2 = st.columns([1, 2])
    with m1:
        st.markdown("### 🛠️ Global Model Eğitimi")
        st.write("Hedef: `is_top3_family_next_24h` (Binary Classifier)")
        
        train_tf = st.selectbox("Eğitim Timeframe", ["15m", "1h", "4h"], index=0)
        use_bootstrap = st.checkbox("Bootstrap Resampling Kullan (Robustness)", value=True)
        
        if st.button("🚀 GLOBAL EĞİTİMİ BAŞLAT", key="global_train_btn"):
            with st.spinner(f"Global {train_tf} Family-Ranker eğitiliyor... Bu işlem dakikalar sürebilir."):
                import subprocess
                try:
                    # Run the global training script as a subprocess to avoid blocking streamlit
                    result = subprocess.run([sys.executable, "scripts/train_global.py", "--timeframes", train_tf], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success(f"✅ {train_tf} Global Family-Ranker başarıyla eğitildi!")
                        st.text_area("Eğitim Logu:", result.stdout, height=200)
                    else:
                        st.error("❌ Eğitim sırasında bir hata oluştu.")
                        st.text_area("Hata Detayı:", result.stderr, height=200)
                except Exception as e:
                    st.error(f"Sistem Hatası: {e}")

        st.write("---")
        st.markdown("### ⚠️ [LEGACY] Per-Coin Training")
        st.warning("Eski tip 'target_hit' optimizasyonu. Sadece test amaçlıdır.")
        if st.checkbox("Legacy Paneli Göster"):
            gun = st.number_input("Öğrenme Verisi Uzunluğu (Gün)", 15, 360, 60)
            hedef = st.selectbox("Optimizasyon Hedefi", ["Maksimum Potansiyel (Regressor - MFE)", "Başarı Oranı (Classifier - Hit Rate)"])
            
            if st.button("🚀 Legacy Eğitimi Başlat"):
                st.warning("Legacy eğitimi başlatılıyor...")
                # ... legacy loop logic ...
   
            for i, coin in enumerate(target_coins):
                progress_bar.progress((i + 1) / len(target_coins))
                status_area.info(f"⚙️ Eğitiliyor: **{coin}** ({i+1}/{len(target_coins)})")
                try:
                    df_raw = fetcher.fetch_historical_ohlcv(coin, timeframe='1h', days=int(gun))
                    if df_raw.empty or len(df_raw) < 100:
                        trained_fail.append(f"{coin} (yetersiz veri)")
                        continue
                    df_feat = build_feature_pipeline(df_raw)
                    
                    # Build ML labels (target_hit, mfe_r)
                    builder = DatasetBuilder()
                    df_labeled = builder.build(df_feat)
                    
                    if df_labeled.empty or 'target_hit' not in df_labeled.columns:
                        trained_fail.append(f"{coin} (label üretilemedi)")
                        continue
                    
                    ml_eng = MLRankingEngine()
                    success = ml_eng.train(df_labeled, symbol=coin)
                    
                    if success:
                        trained_ok.append(coin)
                    else:
                        trained_fail.append(f"{coin} (tek sınıf)")
                except Exception as e:
                    trained_fail.append(f"{coin} ({str(e)[:40]})")
            
            status_area.empty()
            progress_bar.empty()
            if trained_ok:
                st.success(f"✅ {len(trained_ok)} model başarıyla eğitildi: {', '.join(trained_ok)}")
            if trained_fail:
                st.warning(f"⚠️ {len(trained_fail)} coin atlandı: {', '.join(trained_fail)}")
                
    with m2:
        st.markdown("<div class='panel-title'>📋 Eğitilmiş Model Envanteri & Sağlık Durumu</div>", unsafe_allow_html=True)
        
        import joblib, json as _json
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # ── Tüm model metadata'larını oku ──────────────────────────────────────
        meta_files = sorted([f for f in os.listdir(model_dir) if f.startswith('metadata_') and f.endswith('.json')])
        
        if meta_files:
            rows = []
            for mf in meta_files:
                sym = mf.replace('metadata_', '').replace('.json', '')
                try:
                    with open(os.path.join(model_dir, mf)) as f:
                        m = _json.load(f)
                    rows.append({
                        "🪙 Coin": sym,
                        "📅 Eğitim Tarihi": m.get('trained_at', '?'),
                        "📊 Örnek": m.get('n_samples_total', '?'),
                        "📈 Leader AUC": round(m.get('leader_auc', m.get('auc', 0.5)), 3),
                        "📈 Setup AUC": round(m.get('setup_auc', 0.5), 3),
                        "📉 Hold MAE": round(m.get('hold_mae', 0.0), 3),
                        "🕐 Son Tarama": m.get('last_scan_at') or '—',
                    })
                except Exception:
                    rows.append({"🪙 Coin": sym, "📅 Eğitim Tarihi": "okunamadı"})
            
            model_inventory_df = pd.DataFrame(rows)
            st.dataframe(model_inventory_df, use_container_width=True, height=280)
            
            # ── Özet metrikler ─────────────────────────────────────────────────
            n_models = len(rows)
            avg_auc  = pd.DataFrame(rows)['📈 CV AUC'].mean() if '📈 CV AUC' in pd.DataFrame(rows).columns else 0.5
            last_any = max((r.get('🕐 Son Tarama', '') or '' for r in rows), default='—')
            
            ms1, ms2, ms3, ms4 = st.columns(4)
            ms1.metric("Eğitilmiş Model Sayısı", n_models)
            ms2.metric("Ortalama AUC", f"{avg_auc:.3f}")
            ms3.metric("Sızıntı Koruması", "✅ Aktif", "Chronological Split")
            ms4.metric("Son Tarama (Herhangi)", last_any)
        else:
            st.info("Henüz eğitilmiş model yok. Sol panelden 'Eğitimi Başlat' butonunu kullanın.")
        
        st.write("---")
        
        # ── Seçili model için Feature Importance ──────────────────────────────
        st.markdown("<div class='panel-title'>📊 Feature Önemi (Seçili Model)</div>", unsafe_allow_html=True)
        os.makedirs(model_dir, exist_ok=True)
        available_models = [f.replace('feature_importance_','').replace('.pkl','') 
                           for f in os.listdir(model_dir) if 'feature_importance_' in f]
        if not available_models:
            available_models = ["GLOBAL"]
        
        diag_coin = st.selectbox("Hangi Modelin Feature Önemini Göster?", available_models, index=0)
        importance_path = os.path.join(model_dir, f"feature_importance_{diag_coin}.pkl")
        metrics_path    = os.path.join(model_dir, f"metrics_{diag_coin}.pkl")
        
        auc_val, prec_val = 0.50, 0.0
        onem_df = pd.DataFrame()
        
        if os.path.exists(importance_path):
            try:
                imp_data = joblib.load(importance_path)
                onem_df = pd.DataFrame(list(imp_data.items()), columns=['Değişken Adı', 'Önem Derecesi'])
                onem_df = onem_df.sort_values('Önem Derecesi', ascending=False).head(10)
            except: pass
        
        if os.path.exists(metrics_path):
            try:
                md = joblib.load(metrics_path)
                auc_val  = md.get('auc', 0.50)
                prec_val = md.get('precision_60', 0.0)
            except: pass
        
        fi1, fi2 = st.columns(2)
        fi1.metric(f"{diag_coin} — Long AUC", f"{auc_val:.3f}", help="0.5 = random, 1.0 = perfect")
        fi2.metric(f"{diag_coin} — Folds", md.get('wf_folds', 1) if os.path.exists(metrics_path) else "N/A")
        
        if not onem_df.empty:
            st.bar_chart(onem_df.set_index('Değişken Adı'))
        else:
            st.info("Bu model için feature importance verisi yok. Yeniden eğitin.")


# ==========================================
# PANEL 5: SİSTEM & KONFİGÜRASYON
# ==========================================
elif secili_panel == "Sistem & Konfigürasyon":
    st.markdown("<div class='panel-title'>⚙️ Sistem Ayarları (default.yaml)</div>", unsafe_allow_html=True)
    # Fix #5: Pydantic v2 → .model_dump()
    st.json(config.model_dump(), expanded=True)

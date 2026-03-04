"""
╔══════════════════════════════════════════════════════════════════════╗
║  SWING EDGE OPTIMIZER  v2.0                                          ║
║  Walk-forward · Caché JSON · Compatible Jupyter                      ║
╚══════════════════════════════════════════════════════════════════════╝

CAMBIOS v2:
  · Señales por SCORE ≥ umbral (sistema de puntos) en lugar de AND duro
  · Menos condiciones obligatorias → más trades → resultados válidos
  · Relajado: cada indicador suma puntos, no bloquea solo
  · Walk-forward 65/35, caché JSON, sin argparse
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()

CACHE_FILE  = BASE_DIR / "optimal_params.json"
LOG_FILE    = BASE_DIR / "optimization_log.txt"
DATA_PERIOD = "5y"
N_COMBOS    = 1200   # combinaciones aleatorias por activo

# ── Universo ───────────────────────────────────────────────────────
UNIVERSE = {
    # ── Metales preciosos ──────────────────────────────────────────
    "GLDM.PA":  "Amundi Gold Bugs UCIT ETF",
    "IS0E.DE":  "iShares Gold Producers UCIT ETF",
    "SLVR.DE":  "Global X Silver Miners UCIT ETF",
    "VVMX.DE":  "VanEck Rare Earth & Strategic Metals UCIT",
    # ── Nuclear / Uranio ──────────────────────────────────────────
    "URNU.DE":  "Global X Uranium UCIT ETF",
    # ── Tecnología y semiconductores ──────────────────────────────
    "VVSM.DE":  "VanEck Semiconductor UCIT ETF",
    "SEC0.DE":  "iShares MSCI Global Semiconductors UCIT",
    # ── Índices apalancados ───────────────────────────────────────
    "NDXH.PA":  "Amundi Nasdaq 100 EUR Hedge UCIT ETF",
    "LQQ.PA":   "Amundi Nasdaq 100 Daily 2x Lev UCIT",
    "IBCF.DE":  "iShares S&P 500 EUR Hedge UCIT ETF",
    "DBPG.DE":  "Xtrackers S&P 500 2x Leveraged Daily UCIT",
    # ── Sectoriales USA ───────────────────────────────────────────
    "ZPDE.DE":  "SPDR S&P US Energy Select Sector UCIT",
    "ZPDJ.DE":  "SPDR S&P US Industrials Select Sector UCIT",
    # ── Sectoriales Europa ────────────────────────────────────────
    "EXV1.DE":  "iShares STOXX Europe 600 Banks UCIT ETF",
    # ── Emergentes ────────────────────────────────────────────────
    "EMXC.DE":  "Amundi MSCI Emerging ex China UCIT",
    # ── España ────────────────────────────────────────────────────
    "IBEXA.MC": "Amundi IBEX 35 Doble UCIT ETF",
    # ── Japón ─────────────────────────────────────────────────────
    "WTIF.DE":  "WisdomTree Japan Equity EUR Hedged UCIT",
    # ── Defensa ───────────────────────────────────────────────────
    "DFEN.DE":  "VanEck Defense UCIT ETF",
    # ── Acciones USA — momentum alto ─────────────────────────────
    "WPM":      "Wheaton Precious Metals",
    "CCJ":      "Cameco Corp (Uranium)",
    "VST":      "Vistra Energy Corp",
    "AXON":     "Axon Enterprise Inc",
    "SMCI":     "Super Micro Computer Inc",
    "CELH":     "Celsius Holdings Inc",
    "RCL":      "Royal Caribbean Group",
    "MELI":     "MercadoLibre Inc",
    "NIO":      "NIO Inc. ADR (EV China)",
    "OSCR":     "Oscar Health Inc.",
    "BABA":     "Alibaba Group ADR",
    "ASTS":     "AST SpaceMobile Inc.",
    "GME":      "GameStop Corp.",
    "EVGO":     "EVgo Inc.",
    # ── Inversos / VIX (lógica especial) ─────────────────────────
    "DBPK.DE":  "Xtrackers S&P 500 2x Inverse Daily UCIT",
    "2INVE.MC": "Amundi IBEX 35 Doble Inverso -2x UCIT",
    "LVO.MI":   "Amundi S&P 500 VIX Futures Enhanced Roll",
}

# Tickers inversos: señal alcista = precio BAJANDO (todo se invierte)
INVERSE_TICKERS = {"DBPK.DE", "2INVE.MC"}

# VIX: señal = volatilidad subiendo = mercado cayendo = spike de miedo
# Se trata como inverso pero con lógica propia
VIX_TICKERS = {"LVO.MI"}

# ── Espacio de búsqueda ────────────────────────────────────────────
SEARCH = {
    # RSI
    "rsi_p":      [7, 10, 14],
    "rsi_lo":     [45, 48, 50, 52, 55],
    "rsi_hi":     [65, 70, 75, 80],
    # EMAs
    "ema_f":      [8, 10, 12, 15],
    "ema_s":      [20, 25, 30],
    "ema_t":      [50, 100, 200],
    # MACD
    "macd_f":     [10, 12],
    "macd_s":     [24, 26, 28],
    "macd_sig":   [7, 9],
    # Filtros de calidad
    "adx_min":    [15, 18, 22, 25],
    "vol_min":    [0.8, 1.0, 1.1, 1.2],
    "stoch_hi":   [65, 72, 80],
    "dist_max":   [8.0, 12.0, 16.0],
    # Score mínimo para señal (qué tan exigente es el sistema)
    "score_min":  [55, 60, 65, 70],
    # Gestión de trade
    "atr_stop":   [1.5, 1.8, 2.2, 2.6],
    "tp_pct":     [4.0, 6.0, 8.0, 10.0, 12.0],
    "max_days":   [8, 10, 12, 15],
    "trail_atr":  [1.2, 1.5, 2.0],
    "trail_act":  [2.5, 3.5, 5.0],
}

# ── RSI ───────────────────────────────────────────────────────────
def _rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

# ── Indicadores ───────────────────────────────────────────────────
def calc_ind(df, p, invert=False):
    c = df['Close'].squeeze()
    h = df['High'].squeeze()
    l = df['Low'].squeeze()
    v = df['Volume'].squeeze()
    out = {}
    # Invertir serie para ETFs inversos (precio baja cuando el subyacente sube)
    if invert:
        c = (c.iloc[0] * 2) - c   # reflejo: sube cuando el original baja
    out['c']      = c.values.astype(float)
    out['rsi']    = _rsi(c, p['rsi_p']).values
    out['ema_f']  = c.ewm(span=p['ema_f'], adjust=False).mean().values
    out['ema_s']  = c.ewm(span=p['ema_s'], adjust=False).mean().values
    out['ema_t']  = c.ewm(span=p['ema_t'], adjust=False).mean().values
    mf = c.ewm(span=p['macd_f'], adjust=False).mean()
    ms = c.ewm(span=p['macd_s'], adjust=False).mean()
    macd = mf - ms
    out['macd_h'] = (macd - macd.ewm(span=p['macd_sig'], adjust=False).mean()).values
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    out['atr']    = tr.ewm(span=14, adjust=False).mean().values
    pdm = h.diff().clip(lower=0); ndm = (-l.diff()).clip(lower=0)
    pdm[pdm < ndm] = 0; ndm[ndm < pdm] = 0
    atr14 = tr.ewm(span=14, adjust=False).mean()
    pdi = 100*pdm.ewm(span=14, adjust=False).mean()/atr14.replace(0,np.nan)
    ndi = 100*ndm.ewm(span=14, adjust=False).mean()/atr14.replace(0,np.nan)
    dx  = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    out['adx']    = dx.ewm(span=14, adjust=False).mean().values
    out['pdi']    = pdi.values
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    out['stoch']  = (100*(c-lo14)/(hi14-lo14).replace(0,np.nan)).values
    out['vol_r']  = (v/v.rolling(20).mean()).values
    out['dist']   = ((c-c.ewm(span=p['ema_s'],adjust=False).mean())
                     /c.ewm(span=p['ema_s'],adjust=False).mean()*100).values
    ema_fs = c.ewm(span=p['ema_f'],adjust=False).mean()
    out['slope']  = ((ema_fs-ema_fs.shift(3))/ema_fs.shift(3)*100).values
    out['roc5']   = c.pct_change(5).values*100
    out['roc20']  = c.pct_change(20).values*100
    # Bollinger band width para detectar squeeze
    bb_std = c.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    out['bb_w']   = (4*bb_std/bb_mid*100).values
    return out

# ── Sistema de puntuación (0-100) ─────────────────────────────────
def score_bar(ind, i, p):
    """
    Suma puntos por cada señal alcista.
    Más flexible que condiciones AND — basta con score ≥ score_min.
    """
    def g(k): v=ind[k][i]; return float(v) if not np.isnan(v) else 0.0
    sc = 0

    # RSI: 0-20 pts
    rsi = g('rsi')
    if p['rsi_lo'] <= rsi <= p['rsi_hi']:     sc += 15
    elif rsi > p['rsi_lo'] - 5:              sc +=  7

    # Estructura EMA: 0-25 pts
    c = g('c')
    if c > g('ema_t'):                        sc += 10
    if c > g('ema_s'):                        sc +=  8
    if g('ema_f') > g('ema_s'):              sc +=  7

    # MACD: 0-20 pts
    mh = g('macd_h')
    mh1 = float(ind['macd_h'][i-1]) if i > 0 and not np.isnan(ind['macd_h'][i-1]) else 0.0
    if mh > 0 and mh > mh1:                  sc += 20
    elif mh > 0:                             sc += 12
    elif mh > mh1:                           sc +=  5

    # ADX/tendencia: 0-15 pts
    if g('adx') >= p['adx_min'] and g('pdi') > 0: sc += 15
    elif g('adx') >= p['adx_min'] - 5:            sc +=  7

    # Volumen: 0-10 pts
    vr = g('vol_r')
    if vr >= p['vol_min'] * 1.3:             sc += 10
    elif vr >= p['vol_min']:                 sc +=  6
    elif vr >= p['vol_min'] * 0.85:         sc +=  3

    # Stoch (filtro sobrecompra): 0-5 pts
    if g('stoch') <= p['stoch_hi']:          sc +=  5

    # Slope EMA: 0-5 pts
    if g('slope') > 0.2:                     sc +=  5
    elif g('slope') > 0:                    sc +=  2

    # Penalizaciones
    if g('dist') > p['dist_max']:            sc -= 15   # sobreextendido
    if g('dist') < 0:                        sc -= 10   # bajo EMA_S
    if g('roc5') < -5:                       sc -= 12   # caída reciente
    if rsi > p['rsi_hi']:                    sc -= 15   # sobrecomprado
    if mh < 0 and mh < mh1:                 sc -= 10   # MACD deteriorando

    return max(0, min(100, sc))

# ── Señales ───────────────────────────────────────────────────────
def get_signals(ind, p, macro=None, invert_macro=False):
    n = len(ind['c']); sigs = np.zeros(n, dtype=np.int8); last = -6
    for i in range(35, n):
        if i - last < 4: continue
        sc = score_bar(ind, i, p)
        # Para inversos: señal cuando mercado BAJISTA (macro invertida)
        if invert_macro:
            macro_ok = macro is None or not bool(macro[i])
        else:
            macro_ok = macro is None or bool(macro[i])
        if sc >= p['score_min'] and macro_ok:
            sigs[i] = 1; last = i
    return sigs

# ── Backtest rápido ───────────────────────────────────────────────
def run_bt(ind, sigs, p):
    c=ind['c']; atr=ind['atr']; n=len(c)
    trades=[]; in_t=False; ep=sl=tp=pk=0.0; entry_i=0
    for i in range(n):
        price=c[i]
        if np.isnan(price): continue
        a = atr[i] if not np.isnan(atr[i]) else price*0.02
        if not in_t:
            if sigs[i]==1:
                in_t=True; ep=price; pk=price
                sl=price-a*p['atr_stop']; tp=price*(1+p['tp_pct']/100); entry_i=i
        else:
            pk=max(pk,price); held=i-entry_i
            pnl=(price-ep)/ep*100; reason=None
            if price<=sl:    reason="SL"
            elif price>=tp:  reason="TP"
            elif pnl>p['trail_act']:
                if price<pk-a*p['trail_atr']: reason="Trail"
            if reason is None and held>=p['max_days']: reason="T"
            if reason:
                trades.append((ep,price,pnl,held,reason,entry_i,i))
                in_t=False
    return trades

# ── Score compuesto de métricas ───────────────────────────────────
def score_metrics(trades):
    if len(trades) < 4: return None
    pnls = np.array([t[2] for t in trades])
    wins = pnls[pnls>0]; loss = pnls[pnls<=0]
    n=len(pnls); wr=len(wins)/n*100
    pf = abs(wins.sum()/loss.sum()) if loss.sum()!=0 else (99.0 if len(wins)>0 else 0)
    if pf < 0.8: return None  # perdedor claro
    dd = float(np.min(np.cumsum(pnls)-np.maximum.accumulate(np.cumsum(pnls))))
    sh = float(pnls.mean()/pnls.std()*np.sqrt(26)) if pnls.std()>0 else 0
    p3 = float((pnls>=3).mean()*100)
    p5 = float((pnls>=5).mean()*100)
    dd_pen = max(0,(-dd-20))*0.3
    sc = sh*3.0 + pf*2.0 + p3*0.04 + (wr/100)*0.8 - dd_pen
    return dict(n=n,wr=wr,pf=pf,sharpe=sh,dd=dd,p3=p3,p5=p5,
                avg_w=float(wins.mean()) if len(wins) else 0,
                avg_l=float(loss.mean()) if len(loss) else 0,
                total=float(pnls.sum()), score=sc)

# ── Optimizador walk-forward ───────────────────────────────────────
def optimize(ticker, df, macro, n_combos):
    n=len(df); split=int(n*0.65)
    df_is=df.iloc[:split];  mo_is=macro[:split]
    df_oos=df.iloc[split:]; mo_oos=macro[split:]

    is_inverse = ticker in INVERSE_TICKERS
    is_vix     = ticker in VIX_TICKERS
    use_invert = is_inverse or is_vix   # invertir precio y macro

    np.random.seed(42)
    combos=[]
    attempts=0
    while len(combos)<n_combos and attempts<n_combos*6:
        attempts+=1
        p={k:np.random.choice(v).item() for k,v in SEARCH.items()}
        if p['ema_f']>=p['ema_s'] or p['macd_f']>=p['macd_s']: continue
        combos.append(p)

    # ── In-sample ─────────────────────────────────────────────────
    is_res=[]
    for p in combos:
        try:
            ind=calc_ind(df_is,p,use_invert); sigs=get_signals(ind,p,mo_is,use_invert)
            n_sigs = sigs.sum()
            if n_sigs < 3: continue   # muy pocas señales
            trs=run_bt(ind,sigs,p); m=score_metrics(trs)
            if m: is_res.append((m['score'],p,m))
        except Exception: continue

    if not is_res:
        # Diagnóstico: cuántas señales generamos con parámetros laxos
        p_lax = dict(rsi_p=14,rsi_lo=45,rsi_hi=80,ema_f=10,ema_s=20,ema_t=50,
                     macd_f=12,macd_s=26,macd_sig=9,adx_min=15,vol_min=0.8,
                     stoch_hi=80,dist_max=16,score_min=50,atr_stop=2.0,
                     tp_pct=8.0,max_days=12,trail_atr=1.5,trail_act=3.5)
        ind=calc_ind(df_is,p_lax,use_invert); sigs=get_signals(ind,p_lax,mo_is,use_invert)
        print(f"      diagnóstico: parámetros laxos → {sigs.sum()} señales IS")
        return None

    is_res.sort(key=lambda x:-x[0])

    # ── Out-of-sample: validar top 20 ─────────────────────────────
    oos_res=[]
    for sc_is,p,m_is in is_res[:20]:
        try:
            ind=calc_ind(df_oos,p,use_invert); sigs=get_signals(ind,p,mo_oos,use_invert)
            trs=run_bt(ind,sigs,p); m=score_metrics(trs)
            if m:
                deg=max(0,sc_is-m['score'])
                oos_res.append((m['score']-deg*0.2,sc_is,p,m_is,m))
        except Exception: continue

    if not oos_res: return None
    oos_res.sort(key=lambda x:-x[0])
    _,sc_is,bp,m_is,m_oos=oos_res[0]

    return dict(
        ticker=ticker,
        invert=use_invert,
        params={k:(bool(v) if isinstance(v,np.bool_) else v) for k,v in bp.items()},
        score_is=round(sc_is,3), score_oos=round(m_oos['score'],3),
        metrics_is={k:round(float(v),3) for k,v in m_is.items()},
        metrics_oos={k:round(float(v),3) for k,v in m_oos.items()},
        optimized_at=datetime.now().isoformat(), n_combos=len(combos),
    )

# ══════════════════════════════════════════════════════════════════
def main():
    # ── Configuración — edita aquí ────────────────────────────────
    TICKER_SOLO  = None    # None = todos | "SMH" = solo ese ticker
    FORCE_REOPT  = False   # True = re-optimizar aunque exista caché
    N_COMBOS_RUN = N_COMBOS  # 800 combos por activo
    # ─────────────────────────────────────────────────────────────
    class args:
        ticker = TICKER_SOLO
        force  = FORCE_REOPT
        combos = N_COMBOS_RUN

    cache={}
    if CACHE_FILE.exists():
        cache=json.loads(CACHE_FILE.read_text())
        print(f"Caché: {len(cache)} activos ya optimizados\n")

    tickers=[args.ticker] if args.ticker else list(UNIVERSE.keys())

    print("Descargando SPY para filtro macro...")
    spy=yf.download("SPY",period=DATA_PERIOD,auto_adjust=True,progress=False)
    if isinstance(spy.columns,pd.MultiIndex): spy.columns=spy.columns.get_level_values(0)
    spy_c=spy['Close'].squeeze()
    spy_e=spy_c.ewm(span=50,adjust=False).mean()

    log=[]
    for ticker in tickers:
        name=UNIVERSE.get(ticker,ticker)
        print(f"\n{'─'*58}\n  {ticker}  {name}")
        if ticker in cache and not args.force:
            m=cache[ticker].get('metrics_oos',{})
            print(f"  ⏭  Caché ({cache[ticker].get('optimized_at','')[:10]})  "
                  f"Sharpe={m.get('sharpe',0):.2f}  PF={m.get('pf',0):.2f}  "
                  f"WR={m.get('wr',0):.1f}%  ≥3%={m.get('p3',0):.0f}%")
            continue

        print(f"  ↓ Descargando {DATA_PERIOD}...",end=" ")
        try:
            df=yf.download(ticker,period=DATA_PERIOD,auto_adjust=True,progress=False)
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            if len(df)<200: print("❌ pocos datos"); continue
            print(f"✓ {len(df)}d")
        except Exception as e: print(f"❌ {e}"); continue

        macro=(spy_c>spy_e).reindex(df.index,method='ffill').fillna(False).values

        print(f"  ⚙  Optimizando {args.combos} combos...",end=" ",flush=True)
        t0=time.time(); res=optimize(ticker,df,macro,args.combos); dt=time.time()-t0

        if res:
            m=res['metrics_oos']
            print(f"✓ {dt:.0f}s  Sharpe={m['sharpe']:.2f}  PF={m['pf']:.2f}  "
                  f"WR={m['wr']:.1f}%  ≥3%={m['p3']:.0f}%  "
                  f"N={m['n']}  DD={m['dd']:.1f}%")
            cache[ticker]=res
            CACHE_FILE.write_text(json.dumps(cache,indent=2))
            log.append(f"{ticker}: Sharpe={m['sharpe']:.2f} PF={m['pf']:.2f} "
                       f"WR={m['wr']:.1f}% ≥3%={m['p3']:.0f}% Trades={m['n']} DD={m['dd']:.1f}%")
        else:
            print("❌ sin parámetros válidos")

    LOG_FILE.write_text(f"Optimización: {datetime.now()}\n\n"+"\n".join(log))
    print(f"\n{'═'*58}\nCompletado. Caché: {CACHE_FILE}")

if __name__=="__main__":
    main()

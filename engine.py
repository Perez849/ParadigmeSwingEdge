"""
╔══════════════════════════════════════════════════════════════════════╗
║  SWING EDGE ENGINE  v2.0                                             ║
║  Usa EXACTAMENTE las mismas funciones que optimizer.py               ║
║  → Métricas del resumen tomadas del OOS (honest metrics)             ║
╚══════════════════════════════════════════════════════════════════════╝

USO (Jupyter):
  Edita las variables de configuración en main() y ejecuta main()
"""

import sys, json, warnings
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
OUTPUT_FILE = BASE_DIR / "dashboard_data.json"
PERIOD      = "3y"

# Colores terminal
G="\033[92m"; R="\033[91m"; Y="\033[93m"; B="\033[94m"
C="\033[96m"; M="\033[95m"; DIM="\033[90m"; BOLD="\033[1m"; RST="\033[0m"

# ══════════════════════════════════════════════════════════════════
# BLOQUE COMPARTIDO — COPIA EXACTA DE optimizer.py
# ══════════════════════════════════════════════════════════════════

def _rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def calc_ind(df, p, invert=False):
    c = df['Close'].squeeze()
    h = df['High'].squeeze()
    l = df['Low'].squeeze()
    v = df['Volume'].squeeze()
    out = {}
    if invert:
        c = (c.iloc[0] * 2) - c
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
    bb_std = c.rolling(20).std()
    bb_mid = c.rolling(20).mean()
    out['bb_w']   = (4*bb_std/bb_mid*100).values
    # Precio real siempre (sin invertir) — para mostrar en dashboard y alertas
    c_real        = df['Close'].squeeze()
    out['close']  = c_real.values.astype(float)   # precio real
    out['high']   = h.values.astype(float)
    out['low']    = l.values.astype(float)
    out['open']   = df['Open'].squeeze().values.astype(float) if 'Open' in df.columns else out['close']
    out['volume'] = v.values.astype(float)
    out['index']  = df.index
    return out

def score_bar(ind, i, p):
    def g(k): v=ind[k][i]; return float(v) if not np.isnan(v) else 0.0
    sc = 0
    rsi = g('rsi')
    if p['rsi_lo'] <= rsi <= p['rsi_hi']:     sc += 15
    elif rsi > p['rsi_lo'] - 5:              sc +=  7
    c = g('c')
    if c > g('ema_t'):                        sc += 10
    if c > g('ema_s'):                        sc +=  8
    if g('ema_f') > g('ema_s'):              sc +=  7
    mh = g('macd_h')
    mh1 = float(ind['macd_h'][i-1]) if i > 0 and not np.isnan(ind['macd_h'][i-1]) else 0.0
    if mh > 0 and mh > mh1:                  sc += 20
    elif mh > 0:                             sc += 12
    elif mh > mh1:                           sc +=  5
    if g('adx') >= p['adx_min'] and g('pdi') > 0: sc += 15
    elif g('adx') >= p['adx_min'] - 5:            sc +=  7
    vr = g('vol_r')
    if vr >= p['vol_min'] * 1.3:             sc += 10
    elif vr >= p['vol_min']:                 sc +=  6
    elif vr >= p['vol_min'] * 0.85:         sc +=  3
    if g('stoch') <= p['stoch_hi']:          sc +=  5
    if g('slope') > 0.2:                     sc +=  5
    elif g('slope') > 0:                    sc +=  2
    if g('dist') > p['dist_max']:            sc -= 15
    if g('dist') < 0:                        sc -= 10
    if g('roc5') < -5:                       sc -= 12
    if rsi > p['rsi_hi']:                    sc -= 15
    if mh < 0 and mh < mh1:                 sc -= 10
    return max(0, min(100, sc))

def get_signals(ind, p, macro=None, invert_macro=False):
    n = len(ind['c']); sigs = np.zeros(n, dtype=np.int8); last = -6
    for i in range(35, n):
        if i - last < 4: continue
        sc = score_bar(ind, i, p)
        if invert_macro:
            macro_ok = macro is None or not bool(macro[i])
        else:
            macro_ok = macro is None or bool(macro[i])
        if sc >= p['score_min'] and macro_ok:
            sigs[i] = 1; last = i
    return sigs

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

def score_metrics(trades):
    if len(trades) < 4: return None
    pnls = np.array([t[2] for t in trades])
    wins = pnls[pnls>0]; loss = pnls[pnls<=0]
    n=len(pnls); wr=len(wins)/n*100
    pf = abs(wins.sum()/loss.sum()) if loss.sum()!=0 else (99.0 if len(wins)>0 else 0)
    if pf < 0.8: return None
    dd = float(np.min(np.cumsum(pnls)-np.maximum.accumulate(np.cumsum(pnls))))
    sh = float(pnls.mean()/pnls.std()*np.sqrt(26)) if pnls.std()>0 else 0
    p3 = float((pnls>=3).mean()*100)
    p5 = float((pnls>=5).mean()*100)
    dd_pen = max(0,(-dd-20))*0.3
    sc = sh*3.0 + pf*2.0 + p3*0.04 + (wr/100)*0.8 - dd_pen
    return dict(n=n, wr=wr, pf=pf, sharpe=sh, dd=dd, p3=p3, p5=p5,
                avg_w=float(wins.mean()) if len(wins) else 0,
                avg_l=float(loss.mean()) if len(loss) else 0,
                total=float(pnls.sum()), score=sc)

# ══════════════════════════════════════════════════════════════════
# FUNCIONES EXCLUSIVAS DEL ENGINE
# ══════════════════════════════════════════════════════════════════

def build_rich_trades(ind, sigs, p, ticker):
    c=ind['c']; real=ind['close']; atr=ind['atr']; idx=ind['index']; n=len(c)
    trades=[]; in_t=False; ep=sl=tp=pk=0.0; ep_real=0.0; entry_i=0
    open_trade=None  # trade actualmente abierto al final del histórico

    for i in range(n):
        price=c[i]
        price_real=real[i]
        if np.isnan(price): continue
        a = atr[i] if not np.isnan(atr[i]) else abs(price)*0.02

        if not in_t:
            if sigs[i]==1:
                in_t=True; ep=price; ep_real=price_real; pk=price
                sl=price-a*p['atr_stop']
                tp=price*(1+p['tp_pct']/100)
                entry_i=i
        else:
            pk=max(pk,price); held=i-entry_i
            pnl=(price-ep)/ep*100; reason=None
            if price<=sl:    reason="Stop Loss"
            elif price>=tp:  reason="Take Profit ✅"
            elif pnl>p['trail_act']:
                nl = pk - a*p['trail_atr']
                if price < nl:
                    reason="Trailing Stop"
                else:
                    sl = max(sl, nl)  # actualizar trailing SL
            if reason is None and held>=p['max_days']:
                reason=f"Tiempo ({held}d)"
            if reason:
                entry_sc = int(score_bar(ind, entry_i, p))
                sl_real = round(ep_real * (1 + (sl - ep)/ep), 2)
                tp_real = round(ep_real * (1 + p['tp_pct']/100), 2)
                trades.append({
                    "ticker":      ticker,
                    "entry_date":  str(idx[entry_i])[:10],
                    "exit_date":   str(idx[i])[:10],
                    "entry_price": round(ep_real, 2),
                    "exit_price":  round(price_real, 2),
                    "stop_loss":   sl_real,
                    "take_profit": tp_real,
                    "pnl":         round(pnl, 2),
                    "peak_pnl":    round((pk-ep)/ep*100, 2),
                    "days":        held,
                    "reason":      reason,
                    "score":       entry_sc,
                    "entry_rsi":   round(float(ind['rsi'][entry_i]),1) if not np.isnan(ind['rsi'][entry_i]) else None,
                    "entry_adx":   round(float(ind['adx'][entry_i]),1) if not np.isnan(ind['adx'][entry_i]) else None,
                })
                in_t=False

    # Si al terminar el bucle hay un trade abierto, guardarlo con el SL actualizado
    if in_t:
        sl_real = round(ep_real * (1 + (sl - ep)/ep), 2)
        tp_real = round(ep_real * (1 + p['tp_pct']/100), 2)
        open_trade = {
            "entry_date":  str(idx[entry_i])[:10],
            "entry_price": round(ep_real, 2),
            "stop_loss":   sl_real,   # SL actual (puede ser trailing actualizado)
            "take_profit": tp_real,
            "days_open":   n - 1 - entry_i,
        }

    return trades, open_trade


def check_alert(ind, sigs, p, ticker, name, trades=None, open_trade=None):
    today_str = str(ind['index'][-1])[:10]
    today_ts  = pd.Timestamp(today_str)

    # Si hay un trade abierto simulado, es la única fuente de verdad
    if open_trade:
        date_str = open_trade['entry_date']
        price_real = open_trade['entry_price']
        sl_real    = open_trade['stop_loss']
        tp_real    = open_trade['take_profit']
        current_real = float(ind['close'][-1])

        # El trade ya debería haber cerrado por SL
        if current_real <= sl_real:
            return None

        # days_ago en días de MERCADO (barras), no días calendario
        # El optimizador usa max_days en barras — debe ser consistente
        entry_i = next((i for i in range(len(ind['index'])) if str(ind['index'][i])[:10]==date_str), len(ind['index'])-1)
        days_ago = len(ind['index']) - 1 - entry_i  # barras desde entrada hasta hoy

        # Respetar max_days — si ya pasó, el trade debería haber cerrado en el backtest
        # pero por si acaso no mostramos alerta expirada
        if days_ago > p['max_days']:
            return None

        sl_pct_val = abs((sl_real - price_real) / price_real * 100)
        a = float(ind['atr'][entry_i]) if not np.isnan(ind['atr'][entry_i]) else abs(float(ind['c'][entry_i]))*0.02
        price_c = float(ind['c'][entry_i])
        atr_pct = a / abs(price_c) * 100 if price_c != 0 else 0.02

        return {
            "ticker":      ticker,
            "name":        name,
            "date":        date_str,
            "urgency":     "HOY" if days_ago==0 else f"HACE {days_ago}d",
            "days_ago":    days_ago,
            "price":       price_real,
            "stop_loss":   sl_real,
            "take_profit": tp_real,
            "stop_pct":    round(sl_pct_val, 2),
            "tp_pct":      p['tp_pct'],
            "rr_ratio":    round(p['tp_pct']/sl_pct_val, 2) if sl_pct_val>0 else None,
            "score":       int(score_bar(ind, entry_i, p)),
            "rsi":         round(float(ind['rsi'][entry_i]),1) if not np.isnan(ind['rsi'][entry_i]) else None,
            "adx":         round(float(ind['adx'][entry_i]),1) if not np.isnan(ind['adx'][entry_i]) else None,
            "vol_ratio":   round(float(ind['vol_r'][entry_i]),2) if not np.isnan(ind['vol_r'][entry_i]) else None,
            "max_days":    p['max_days'],
            "trail_act":   p['trail_act'],
            "trail_atr":   p['trail_atr'],
            "trail_sl":    round(price_real * (1 - p['trail_atr']*atr_pct/100), 2),
        }

    # Sin trade abierto simulado → no hay alerta activa
    return None

def build_price_history(ind, sigs, scores):
    n = len(ind['c'])
    start = max(0, n-90)
    hist = []
    for i in range(start, n):
        def gv(k): v=ind[k][i]; return round(float(v),2) if not np.isnan(v) else None
        hist.append({
            "date":   str(ind['index'][i])[:10],
            "open":   gv('open'),  "high":  gv('high'),
            "low":    gv('low'),   "close": gv('close'),  # precio real
            "volume": int(ind['volume'][i]) if not np.isnan(ind['volume'][i]) else 0,
            "ema_f":  gv('ema_f'), "ema_s": gv('ema_s'), "ema_t": gv('ema_t'),
            "rsi":    round(float(ind['rsi'][i]),1) if not np.isnan(ind['rsi'][i]) else None,
            "adx":    round(float(ind['adx'][i]),1) if not np.isnan(ind['adx'][i]) else None,
            "macd_h": round(float(ind['macd_h'][i]),3) if not np.isnan(ind['macd_h'][i]) else None,
            "vol_r":  round(float(ind['vol_r'][i]),2) if not np.isnan(ind['vol_r'][i]) else None,
            "signal": int(sigs[i]),
            "score":  int(scores[i]),
        })
    return hist

# ══════════════════════════════════════════════════════════════════
# TERMINAL OUTPUT
# ══════════════════════════════════════════════════════════════════

def print_summary(ticker, name, m, p):
    """m son las métricas OOS del optimizer — las únicas honestas."""
    col = G if m.get('total',0)>0 else R
    print(f"\n{BOLD}  {'─'*62}{RST}")
    print(f"  {C}{BOLD}{ticker}{RST}  {DIM}{name}{RST}")
    rows = [
        ("Trades (OOS)",    f"{int(m.get('n',0))}",                         True),
        ("Win Rate (OOS)",  f"{m.get('wr',0):.1f}%",                        m.get('wr',0)>=50),
        ("Profit Factor",   f"{m.get('pf',0):.2f}",                         m.get('pf',0)>=1.5),
        ("Avg Win / Loss",  f"+{m.get('avg_w',0):.2f}% / {m.get('avg_l',0):.2f}%", m.get('avg_w',0)>=3),
        ("Sharpe",          f"{m.get('sharpe',0):.2f}",                     m.get('sharpe',0)>=1.0),
        ("Max Drawdown",    f"{m.get('dd',0):.1f}%",                        m.get('dd',0)>=-20),
        ("≥3% / ≥5%",      f"{m.get('p3',0):.0f}% / {m.get('p5',0):.0f}%",m.get('p3',0)>=30),
        ("Retorno OOS",     f"{col}{m.get('total',0):+.1f}%{RST}",          m.get('total',0)>0),
    ]
    for label, val, good in rows:
        ic = G if good else Y
        print(f"  {ic}{'✓' if good else '⚠'}{RST}  {label:<22} {BOLD}{val}{RST}")
    print(f"\n  {DIM}score_min={p['score_min']} rsi={p['rsi_p']}p "
          f"ema={p['ema_f']}/{p['ema_s']}/{p['ema_t']} "
          f"tp={p['tp_pct']}% sl={p['atr_stop']}×ATR "
          f"adx≥{p['adx_min']} max={p['max_days']}d{RST}")

def print_trades(trades, ticker):
    print(f"\n  {DIM}{'─'*74}{RST}")
    print(f"  {BOLD}Historial completo — {ticker}{RST}")
    print(f"  {DIM}{'─'*74}{RST}")
    print(f"  {'Entrada':<12}{'Salida':<12}{'Ent$':>8}{'Sal$':>8}{'P&L':>9}  {'d':>4}{'Sc':>4}  Razón")
    print(f"  {DIM}{'─'*74}{RST}")
    for t in trades:
        p = t['pnl']
        col = G if p>=0 else R
        stars = "★★" if p>=5 else ("★ " if p>=3 else "  ")
        print(f"  {t['entry_date']:<12}{t['exit_date']:<12}"
              f"{t['entry_price']:>8.2f}{t['exit_price']:>8.2f}"
              f"  {col}{p:>+7.2f}%{RST} {col}{stars}{RST}"
              f"  {t['days']:>4}{t['score']:>4}  {DIM}{t['reason']}{RST}")

def print_alert(a):
    col = M if a['days_ago']==0 else Y
    rr  = a.get('rr_ratio', '?')
    rr_col = G if (rr != '?' and rr >= 1.5) else (Y if rr != '?' and rr >= 1.0 else R)
    print(f"\n  {BOLD}🚨 {M}{a['ticker']}{RST} {DIM}{a['name']}{RST}")
    print(f"  {col}{BOLD}{a['urgency']}{RST}  {a['date']}  "
          f"${a['price']:.2f}  Score {G}{a['score']}/100{RST}")
    print(f"  SL fijo:        {R}${a['stop_loss']:.2f} (-{a['stop_pct']}%){RST}")
    print(f"  TP objetivo:    {G}${a['take_profit']:.2f} (+{a['tp_pct']}%){RST}")
    print(f"  Trailing stop:  activa tras +{a.get('trail_act','?')}% → "
          f"stop a {a.get('trail_atr','?')}×ATR del máximo")
    print(f"  Salida por tiempo: máx {a.get('max_days','?')} días hábiles")
    print(f"  R/R: {rr_col}{BOLD}{rr}:1{RST}  "
          f"{DIM}({'bueno ✓' if rr != '?' and rr >= 1.5 else 'ajustado ⚠' if rr != '?' and rr >= 1.0 else 'bajo ✗'}){RST}")

# ══════════════════════════════════════════════════════════════════
# UNIVERSE
# ══════════════════════════════════════════════════════════════════

UNIVERSE = {
    "GLDM.PA":  "Amundi Gold Bugs UCIT ETF",
    "IS0E.DE":  "iShares Gold Producers UCIT ETF",
    "SLVR.DE":  "Global X Silver Miners UCIT ETF",
    "VVMX.DE":  "VanEck Rare Earth & Strategic Metals UCIT",
    "URNU.DE":  "Global X Uranium UCIT ETF",
    "VVSM.DE":  "VanEck Semiconductor UCIT ETF",
    "SEC0.DE":  "iShares MSCI Global Semiconductors UCIT",
    "NDXH.PA":  "Amundi Nasdaq 100 EUR Hedge UCIT ETF",
    "LQQ.PA":   "Amundi Nasdaq 100 Daily 2x Lev UCIT",
    "IBCF.DE":  "iShares S&P 500 EUR Hedge UCIT ETF",
    "DBPG.DE":  "Xtrackers S&P 500 2x Leveraged Daily UCIT",
    "ZPDE.DE":  "SPDR S&P US Energy Select Sector UCIT",
    "ZPDJ.DE":  "SPDR S&P US Industrials Select Sector UCIT",
    "EXV1.DE":  "iShares STOXX Europe 600 Banks UCIT ETF",
    "EMXC.DE":  "Amundi MSCI Emerging ex China UCIT",
    "IBEXA.MC": "Amundi IBEX 35 Doble UCIT ETF",
    "WTIF.DE":  "WisdomTree Japan Equity EUR Hedged UCIT",
    "DFEN.DE":  "VanEck Defense UCIT ETF",
    "WPM":      "Wheaton Precious Metals",
    "CCJ":      "Cameco Corp (Uranium)",
    "VST":      "Vistra Energy Corp",
    "AXON":     "Axon Enterprise Inc",
    "SMCI":     "Super Micro Computer Inc",
    "CELH":     "Celsius Holdings Inc",
    "RCL":      "Royal Caribbean Group",
    "F":        "Ford Motor Company",
    "ENR.DE":   "Siemens Energy AG",
    "KRW.PA":   "Amundi MSCI Korea UCITS ETF",
    "NIO":      "NIO Inc. ADR (EV China)",
    "OSCR":     "Oscar Health Inc.",
    "BABA":     "Alibaba Group ADR",
    "ASTS":     "AST SpaceMobile Inc.",
    "EVGO":     "EVgo Inc.",
    "DBPK.DE":  "Xtrackers S&P 500 2x Inverse Daily UCIT",
    "2INVE.MC": "Amundi IBEX 35 Doble Inverso -2x UCIT",
    "LVO.MI":   "Amundi S&P 500 VIX Futures Enhanced Roll",
}

INVERSE_TICKERS = {"DBPK.DE", "2INVE.MC"}
VIX_TICKERS     = {"LVO.MI"}

# ══════════════════════════════════════════════════════════════════
# OPCIONES — MICROESTRUCTURA DE MERCADO
# ══════════════════════════════════════════════════════════════════

OPTIONS_PROXY = {
    "GLDM.PA":"GLD","IS0E.DE":"GDX","SLVR.DE":"SLV","VVMX.DE":"REMX",
    "URNU.DE":"URA","VVSM.DE":"SMH","SEC0.DE":"SMH",
    "NDXH.PA":"QQQ","LQQ.PA":"QQQ","IBCF.DE":"SPY","DBPG.DE":"SPY",
    "DBPK.DE":"SPY","2INVE.MC":"EWP","ZPDE.DE":"XLE","ZPDJ.DE":"XLI",
    "EXV1.DE":"EUFN","IBEXA.MC":"EWP","EMXC.DE":"EEM","WTIF.DE":"EWJ",
    "DFEN.DE":"ITA","KRW.PA":"EWY","LVO.MI":"VXX","ENR.DE":"XLE",
}

def fmtOI_py(n):
    if n is None or n == 0: return "0"
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M".rstrip('0').rstrip('.')+'M' if '.' in f"{n/1_000_000:.1f}" else f"{n/1_000_000:.0f}M"
    if n >= 1_000: return f"{n/1_000:.1f}K"
    return str(n)

def fetch_options_data(ticker, entry_price, take_profit, stop_loss, days_remaining=None):
    """Obtiene microestructura de opciones con validaciones de calidad de datos."""

    # Proxies donde el PCR alto es ESTRUCTURAL (no señal bajista)
    PCR_STRUCTURAL_HIGH = {
        "GDX":  "ETF de mineras de oro — institucionales cubren largos en oro con puts de GDX. PCR >2 es normal.",
        "GLD":  "ETF de oro físico — cobertura institucional masiva. PCR elevado es estructural.",
        "SLV":  "ETF de plata — igual que GLD, PCR alto es normal por cobertura.",
        "URA":  "ETF de uranio — sector nicho con cobertura institucional. PCR estructuralmente alto.",
        "REMX": "ETF de metales raros — mercado pequeño, PCR poco representativo.",
        "EEM":  "ETF mercados emergentes — cobertura macro institucional. PCR >2 habitual.",
        "EWY":  "ETF Corea — cobertura macro. PCR elevado estructural.",
        "EWP":  "ETF España/IBEX — mercado pequeño, PCR poco representativo.",
        "VXX":  "ETF de volatilidad — por definición se compran puts para cubrirse. PCR siempre alto.",
        "EUFN": "ETF bancos europeos — cobertura institucional. PCR elevado normal.",
        "EWJ":  "ETF Japón — cobertura macro. PCR estructuralmente alto.",
        "XLE":  "ETF energía — cobertura de productores con puts. PCR >1.5 estructural.",
        "XLI":  "ETF industriales — cobertura institucional. PCR moderadamente alto es normal.",
        "ITA":  "ETF defensa — sector concentrado, PCR poco representativo.",
        "SMH":  "ETF semiconductores — muy usado para cubrir posiciones tech. PCR >1.5 normal.",
    }

    def safe_float(val, default=0.0):
        try:
            v = float(val)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    def clean_chain(df):
        for col in ['openInterest','volume','bid','ask','lastPrice','impliedVolatility']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: safe_float(x, 0.0))
        return df

    def best_price(row, col_bid, col_ask, col_last):
        """Precio más fiable: midpoint si spread razonable, si no lastPrice."""
        b = safe_float(row.get(col_bid, 0))
        a = safe_float(row.get(col_ask, 0))
        lp = safe_float(row.get(col_last, 0))
        if b > 0 and a > 0 and a >= b:
            spread_pct = (a - b) / ((a + b) / 2) * 100
            if spread_pct < 50:  # spread razonable (<50%)
                return (b + a) / 2
        return lp if lp > 0 else 0.0

    def pick_best_expiration(expirations, today, min_days=1):
        """Siempre elige el vencimiento más líquido (20-45 días).
        La IV se extrae de ahí y luego se escala al horizonte que se necesite.
        Si no hay entre 20-45d, coge el más cercano disponible ≥ min_days."""
        from datetime import datetime as dt
        candidates = []
        for exp in expirations:
            d = (dt.strptime(exp, "%Y-%m-%d") - today).days
            if d >= min_days:
                candidates.append((exp, d))
            if len(candidates) >= 8:
                break
        if not candidates:
            return None, 0
        # Preferir el rango más líquido
        preferred = [(e, d) for e, d in candidates if 20 <= d <= 45]
        if preferred:
            return preferred[0]
        # Si no hay, el más cercano disponible
        return candidates[0]

    try:
        from datetime import datetime as dt
        opt_ticker = OPTIONS_PROXY.get(ticker, ticker)
        is_proxy   = opt_ticker != ticker

        t = yf.Ticker(opt_ticker)
        expirations = t.options
        if not expirations:
            return {"error": f"Sin opciones para {opt_ticker}"}

        today = dt.now()
        next_exp, days_to_exp = pick_best_expiration(expirations, today)
        if not next_exp:
            return {"error": "Sin vencimientos válidos"}

        chain = t.option_chain(next_exp)
        calls = clean_chain(chain.calls.copy())
        puts  = clean_chain(chain.puts.copy())

        # Precio actual
        hist = t.history(period="5d")
        if hist.empty:
            return {"error": "Sin precio actual"}
        proxy_price = float(hist['Close'].iloc[-1])
        if proxy_price <= 0:
            return {"error": "Precio inválido"}

        scale = entry_price / proxy_price if (is_proxy and entry_price and proxy_price > 0) else 1.0

        # Filtrar strikes ±25% (más estricto — strips OTM extremo con datos basura)
        lo, hi = proxy_price * 0.75, proxy_price * 1.25
        calls = calls[(calls['strike'] >= lo) & (calls['strike'] <= hi)].copy()
        puts  = puts[(puts['strike'] >= lo) & (puts['strike'] <= hi)].copy()
        if calls.empty or puts.empty:
            return {"error": "Sin strikes relevantes en rango ±25%"}

        # ── PUT/CALL RATIO ────────────────────────────────────────────
        # Usar volumen como fallback si OI es 0 en más del 80% de strikes
        c_oi_raw = calls['openInterest'].sum()
        p_oi_raw = puts['openInterest'].sum()
        c_vol    = calls['volume'].sum()
        p_vol    = puts['volume'].sum()

        oi_coverage = (calls['openInterest'] > 0).mean()  # % strikes con OI real
        use_volume_fallback = oi_coverage < 0.2  # menos del 20% tiene OI → usar volumen

        if use_volume_fallback and c_vol > 0:
            c_oi = c_vol; p_oi = p_vol
            pcr_source = "volume"
        else:
            c_oi = c_oi_raw; p_oi = p_oi_raw
            pcr_source = "oi"

        # Validación de sanidad: PCR entre 0.05 y 10 es razonable
        if c_oi > 0:
            raw_pcr = p_oi / c_oi
            pcr = round(raw_pcr, 2) if 0.05 <= raw_pcr <= 10 else None
            pcr_note = "datos insuficientes (ratio anómalo)" if pcr is None else pcr_source
        else:
            pcr = None
            pcr_note = "sin datos de OI/volumen"

        # ── MAX PAIN ─────────────────────────────────────────────────
        # Solo calcular si hay OI real suficiente
        mp_strike = proxy_price
        mp_price  = round(proxy_price * scale, 2)
        mp_pct    = 0.0
        mp_valid  = False
        if c_oi_raw > 0 and p_oi_raw > 0:
            strikes_all = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
            pain = {}
            for s in strikes_all:
                c_itm = calls[calls['strike'] < s]
                p_itm = puts[puts['strike'] > s]
                c_loss = ((s - c_itm['strike']) * c_itm['openInterest']).sum()
                p_loss = ((p_itm['strike'] - s) * p_itm['openInterest']).sum()
                pain[s] = c_loss + p_loss
            if pain:
                mp_strike = min(pain, key=pain.get)
                mp_price  = round(mp_strike * scale, 2)
                mp_pct    = round((mp_strike / proxy_price - 1) * 100, 1)
                mp_valid  = True

        # ── IMPLIED MOVE ─────────────────────────────────────────────
        # Paso 1: extraer precio del straddle ATM del vencimiento líquido
        # Paso 2: destilar IV anualizada
        # Paso 3: escalar a los días restantes del trade (no al vencimiento)
        strikes_all = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        atm_candidates = [s for s in strikes_all if abs(s - proxy_price) / proxy_price < 0.02]
        if not atm_candidates:
            atm_candidates = [min(strikes_all, key=lambda x: abs(x - proxy_price))]

        straddle_pct = None   # precio straddle como % del subyacente
        atm_iv_ann   = None   # IV anualizada extraída del straddle
        impl_pct     = None   # implied move escalado a días_restantes del trade
        atm_used     = None

        for atm_s in sorted(atm_candidates, key=lambda x: abs(x - proxy_price)):
            ac = calls[calls['strike'] == atm_s]
            ap = puts[puts['strike'] == atm_s]
            if ac.empty or ap.empty:
                continue
            cp = best_price(ac.iloc[0], 'bid', 'ask', 'lastPrice')
            pp = best_price(ap.iloc[0], 'bid', 'ask', 'lastPrice')
            if cp > 0 and pp > 0:
                raw_straddle = (cp + pp) / proxy_price * 100
                if 0.5 <= raw_straddle <= 50:
                    straddle_pct = round(raw_straddle, 1)
                    atm_used = atm_s
                    # IV anualizada: straddle/S = IV * sqrt(T) * sqrt(2/pi)
                    # → IV = (straddle/S) / sqrt(T) / sqrt(2/pi) = (straddle/S) * sqrt(pi/2) / sqrt(T)
                    if days_to_exp and days_to_exp > 0:
                        T_exp = days_to_exp / 365
                        atm_iv_ann = straddle_pct / 100 * np.sqrt(np.pi / 2) / np.sqrt(T_exp) * 100
                        if not (5 <= atm_iv_ann <= 300):
                            atm_iv_ann = None
                    break

        # Escalar IV anualizada a los días restantes del trade
        if atm_iv_ann and days_remaining and days_remaining > 0:
            T_trade = days_remaining / 365
            impl_pct = round(atm_iv_ann / 100 / np.sqrt(np.pi / 2) * np.sqrt(T_trade) * 100, 1)
        elif straddle_pct:
            # Fallback: si no hay days_remaining, usar el straddle directo del vencimiento
            impl_pct = straddle_pct

        impl_up     = round(entry_price * (1 + impl_pct / 100), 2) if impl_pct and entry_price else None
        impl_down   = round(entry_price * (1 - impl_pct / 100), 2) if impl_pct and entry_price else None
        tp_in_range = (take_profit <= impl_up) if (impl_up and take_profit) else None

        # ── SKEW desde precios de opciones (no desde impliedVolatility) ──
        # Comparar precio de puts OTM vs calls OTM equidistantes del ATM
        # Si puts OTM cuestan más que calls OTM → skew positivo (miedo bajista)
        skew = None
        skew_detail = {}
        try:
            # Buscar calls y puts ~5% OTM
            strike_otm_c = proxy_price * 1.05
            strike_otm_p = proxy_price * 0.95
            # El strike más cercano al 5% OTM
            best_c = min(calls['strike'].tolist(), key=lambda x: abs(x - strike_otm_c))
            best_p = min(puts['strike'].tolist(),  key=lambda x: abs(x - strike_otm_p))
            # Solo si están realmente ~3-8% OTM
            if 0.02 < abs(best_c/proxy_price - 1) < 0.10 and 0.02 < abs(best_p/proxy_price - 1) < 0.10:
                ac_otm = calls[calls['strike'] == best_c]
                ap_otm = puts[puts['strike'] == best_p]
                if not ac_otm.empty and not ap_otm.empty:
                    cp_otm = best_price(ac_otm.iloc[0], 'bid', 'ask', 'lastPrice')
                    pp_otm = best_price(ap_otm.iloc[0], 'bid', 'ask', 'lastPrice')
                    if cp_otm > 0 and pp_otm > 0:
                        # Normalizar por distancia al strike (comparación justa)
                        c_dist = abs(best_c - proxy_price)
                        p_dist = abs(best_p - proxy_price)
                        cp_norm = cp_otm / c_dist if c_dist > 0 else 0
                        pp_norm = pp_otm / p_dist if p_dist > 0 else 0
                        if cp_norm > 0:
                            raw_skew = (pp_norm - cp_norm) / cp_norm * 100
                            if -80 <= raw_skew <= 300:
                                skew = round(raw_skew, 1)
                                skew_detail = {
                                    "put_strike": round(best_p * scale, 2),
                                    "call_strike": round(best_c * scale, 2),
                                    "put_price":  round(pp_otm, 2),
                                    "call_price": round(cp_otm, 2),
                                }
        except Exception:
            pass

        # ── IV RANK ───────────────────────────────────────────────────
        iv_rank = None
        atm_iv  = round(atm_iv_ann, 1) if atm_iv_ann else None
        hv_252  = None
        hv_21   = None
        try:
            hist_1y = t.history(period="1y")['Close']
            if len(hist_1y) >= 60:
                returns = hist_1y.pct_change().dropna()
                hv_252  = round(returns.std() * np.sqrt(252) * 100, 1)
                hv_21   = round(returns.tail(21).std() * np.sqrt(252) * 100, 1)

                # IV Rank: posición de atm_iv en el rango histórico de HV rolling
                iv_for_rank = atm_iv or hv_21  # fallback a HV21 si no hay IV de opciones
                if iv_for_rank and hv_252 > 0:
                    rolling_hv = returns.rolling(21).std() * np.sqrt(252) * 100
                    iv_min = rolling_hv.quantile(0.05)
                    iv_max = rolling_hv.quantile(0.95)
                    if iv_max > iv_min:
                        iv_rank = round((iv_for_rank - iv_min) / (iv_max - iv_min) * 100, 0)
                        iv_rank = float(max(0, min(100, iv_rank)))
        except Exception:
            pass

        # ── OI MAP ────────────────────────────────────────────────────
        # Usar OI si disponible, volumen como fallback
        def get_oi_col(df):
            if df['openInterest'].sum() > 0:
                return 'openInterest'
            return 'volume'

        c_col = get_oi_col(calls)
        p_col = get_oi_col(puts)
        c_oi_df = calls[calls[c_col] > 0][['strike', c_col]].rename(columns={c_col: 'call_oi'})
        p_oi_df = puts[puts[p_col] > 0][['strike', p_col]].rename(columns={p_col: 'put_oi'})
        oi_map  = pd.merge(c_oi_df, p_oi_df, on='strike', how='outer').fillna(0)
        oi_map['total_oi'] = oi_map['call_oi'] + oi_map['put_oi']
        oi_map = oi_map.nlargest(15, 'total_oi').sort_values('strike')
        oi_using_volume = (c_col == 'volume' or p_col == 'volume')

        oi_list = [{"strike_proxy": round(float(r['strike']), 2),
                    "strike_asset": round(float(r['strike']) * scale, 2),
                    "call_oi": int(r['call_oi']), "put_oi": int(r['put_oi']),
                    "total_oi": int(r['total_oi']),
                    "dominant": "CALL" if r['call_oi'] > r['put_oi'] else "PUT"}
                   for _, r in oi_map.iterrows()]

        # ── GAMMA WALLS (muros de gamma) ─────────────────────────────
        # Strikes con OI > 2x la media — son muros reales de soporte/resistencia
        if oi_list:
            avg_oi = np.mean([x['total_oi'] for x in oi_list])
            gamma_walls = [x for x in oi_list if x['total_oi'] > avg_oi * 2]
            call_walls = sorted([x for x in gamma_walls if x['dominant'] == 'CALL'],
                                key=lambda x: x['total_oi'], reverse=True)[:2]
            put_walls  = sorted([x for x in gamma_walls if x['dominant'] == 'PUT'],
                                key=lambda x: x['total_oi'], reverse=True)[:2]
        else:
            call_walls = []; put_walls = []

        # ── SEÑALES ENRIQUECIDAS ──────────────────────────────────────
        signals = []

        # PCR — con detección de proxies estructuralmente altos
        pcr_structural_note = PCR_STRUCTURAL_HIGH.get(opt_ticker)
        if pcr is not None:
            if pcr_structural_note and pcr > 1.5:
                # PCR alto estructural — no interpretar como señal bajista
                signals.append({"type": "neutral", "msg": f"PCR {pcr} — ALTO ESTRUCTURAL para {opt_ticker}. {pcr_structural_note} No interpretar como señal bajista — es la naturaleza de este mercado. El skew y el implied move son más relevantes aquí."})
            elif pcr > 1.5:
                signals.append({"type": "bearish", "msg": f"PCR {pcr} (alto) — hay {round(p_oi/max(c_oi,1),1)}x más puts que calls. El mercado está pagando por protección bajista. Si el PCR sube más puede ser señal contrarian (todo el mundo ya cubierto = posible suelo), pero de momento confirma presión vendedora."})
            elif pcr > 1.2:
                signals.append({"type": "bearish", "msg": f"PCR {pcr} — más puts que calls. Los inversores están comprando cobertura bajista, lo que sugiere cautela en el mercado. No es extremo pero es moderadamente bajista."})
            elif pcr < 0.6:
                signals.append({"type": "bullish", "msg": f"PCR {pcr} — pocas puts relativas a calls. El mercado está posicionado alcista, sin miedo visible. Confirma la señal técnica de compra."})
            elif pcr < 0.8:
                signals.append({"type": "bullish", "msg": f"PCR {pcr} — leve sesgo alcista. El mercado no está comprando protección, coherente con una señal de compra."})
            else:
                signals.append({"type": "neutral", "msg": f"PCR {pcr} — neutro (rango normal 0.8-1.2). Posicionamiento equilibrado entre calls y puts, sin sesgo direccional claro."})
        else:
            signals.append({"type": "neutral", "msg": f"PCR no disponible ({pcr_note}). Sin datos de posicionamiento en opciones."})

        # Skew
        if skew is not None:
            put_s = skew_detail.get('put_strike', '?')
            call_s = skew_detail.get('call_strike', '?')
            pp = skew_detail.get('put_price', '?')
            cp = skew_detail.get('call_price', '?')
            if skew > 25:
                signals.append({"type": "bearish", "msg": f"Skew muy pronunciado +{skew}% — las puts OTM ({put_s}, precio {pp}) cuestan bastante más que las calls OTM ({call_s}, precio {cp}). El mercado está pagando fuerte por protección bajista. Señal de miedo real o anticipación de caída."})
            elif skew > 10:
                signals.append({"type": "bearish", "msg": f"Skew moderado +{skew}% — puts OTM ({put_s}) más caras que calls OTM ({call_s}). Situación habitual: los inversores pagan más por cobertura que por apalancamiento alcista. No es alarma pero confirma sesgo defensivo."})
            elif skew < -10:
                signals.append({"type": "bullish", "msg": f"Skew invertido {skew}% — calls OTM ({call_s}, precio {cp}) más caras que puts OTM ({put_s}, precio {pp}). El mercado está pagando por participar al alza. Señal de momentum alcista o anticipación de movimiento fuerte."})
            else:
                signals.append({"type": "neutral", "msg": f"Skew equilibrado {skew}% — puts OTM y calls OTM tienen precio similar. Sin sesgo direccional desde opciones."})
        elif skew_detail.get('note'):
            signals.append({"type": "neutral", "msg": f"Skew no disponible: {skew_detail['note']}. Usa los demás indicadores."})
        # Implied Move
        if impl_pct is not None:
            tp_pct_real = round((take_profit / entry_price - 1) * 100, 1) if entry_price else 0
            sl_pct_real = round((1 - stop_loss / entry_price) * 100, 1) if entry_price else 0
            horizon = days_remaining if days_remaining else days_to_exp
            iv_str  = f" (IV anualizada: {round(atm_iv_ann,1)}%)" if atm_iv_ann else ""
            if tp_in_range:
                signals.append({"type": "bullish", "msg": f"Tu TP (+{tp_pct_real}%) está dentro del implied move ±{impl_pct}% calculado para los {horizon}d restantes del trade{iv_str}. El mercado descuenta que ese movimiento es alcanzable. Tu SL (-{sl_pct_real}%) también está dentro del rango — el trade tiene sentido temporalmente."})
            else:
                signals.append({"type": "warning", "msg": f"Tu TP (+{tp_pct_real}%) está FUERA del implied move ±{impl_pct}% para los {horizon}d restantes{iv_str}. El mercado solo descuenta ±{impl_pct}% en ese plazo. Considera ajustar el TP o ampliar el horizonte temporal."})

        # Max Pain
        if mp_valid:
            mp_dist = abs(mp_pct)
            if mp_pct > 3:
                signals.append({"type": "bullish", "msg": f"Max Pain en {mp_price} (+{mp_pct}% sobre precio actual). Los market makers tienen interés en que el precio suba hacia ese nivel antes del vencimiento — actúa como imán alcista. No es garantía pero es una fuerza real en el mercado."})
            elif mp_pct < -3:
                signals.append({"type": "bearish", "msg": f"Max Pain en {mp_price} ({mp_pct}% bajo precio actual). El precio tiende a gravitar hacia Max Pain antes del vencimiento — en este caso implica presión bajista hacia {mp_price}."})
            else:
                signals.append({"type": "neutral", "msg": f"Max Pain en {mp_price} ({mp_pct:+.1f}% vs precio actual) — muy cerca del precio actual. Sin presión direccional significativa desde este ángulo."})

        # IV Rank
        if iv_rank is not None:
            if iv_rank > 70:
                signals.append({"type": "warning", "msg": f"IV Rank {iv_rank:.0f}% — la volatilidad implícita está en el percentil {iv_rank:.0f} de su rango anual. Las opciones están CARAS. Comprar opciones aquí es caro — si vas a operar opciones, mejor vender premium que comprarlo."})
            elif iv_rank < 30:
                signals.append({"type": "bullish", "msg": f"IV Rank {iv_rank:.0f}% — volatilidad implícita BAJA respecto al año. Las opciones están baratas. Buen momento para comprar calls si quieres apalancamiento limitado en riesgo."})
            else:
                signals.append({"type": "neutral", "msg": f"IV Rank {iv_rank:.0f}% — volatilidad en rango normal. Ni cara ni barata."})

        # Gamma Walls
        if call_walls:
            cw_str = ", ".join([f"{w['strike_asset']:.2f} (OI {fmtOI_py(w['call_oi'])})" for w in call_walls])
            signals.append({"type": "neutral", "msg": f"Muros de calls (resistencia gamma): {cw_str}. Concentración alta de calls vendidas — los market makers deben vender el subyacente si el precio sube hacia esos strikes (fuerza natural de resistencia)."})
        if put_walls:
            pw_str = ", ".join([f"{w['strike_asset']:.2f} (OI {fmtOI_py(w['put_oi'])})" for w in put_walls])
            signals.append({"type": "neutral", "msg": f"Muros de puts (soporte gamma): {pw_str}. Concentración alta de puts vendidas — los market makers deben comprar el subyacente si el precio cae hacia esos strikes (fuerza natural de soporte)."})

        # ── VEREDICTO PONDERADO ───────────────────────────────────────
        bulls = sum(1 for s in signals if s['type'] == 'bullish')
        bears = sum(1 for s in signals if s['type'] == 'bearish')
        warns = sum(1 for s in signals if s['type'] == 'warning')
        data_quality = "alta" if pcr is not None and impl_pct is not None and skew is not None else \
                       "media" if sum([pcr is not None, impl_pct is not None, skew is not None]) >= 2 else "baja"

        if data_quality == "baja":
            verdict, vc = "DATOS INSUF.", "yellow"
            vm = "Datos de opciones insuficientes para dar un veredicto fiable. La señal técnica manda."
        elif bulls > bears + warns:
            verdict, vc = "CONFIRMA", "green"
            vm = f"Opciones confirman sesgo alcista ({bulls} señales positivas vs {bears} negativas). Convicción alta."
        elif bears > bulls:
            verdict, vc = "CONTRADICE", "red"
            vm = f"Opciones muestran sesgo bajista ({bears} señales negativas). Considera reducir tamaño o esperar confirmación adicional."
        elif warns > 0 and bulls >= bears:
            verdict, vc = "CONFIRMA CON CAUTELA", "yellow"
            vm = f"Dirección confirmada pero con advertencias — principalmente que tu TP puede ser ambicioso para el vencimiento actual."
        else:
            verdict, vc = "NEUTRO", "yellow"
            vm = "Opciones no dan señal clara. La técnica manda — las opciones no contradicen la entrada."

        return {
            "options_ticker":    opt_ticker,
            "is_proxy":          is_proxy,
            "expiration":        next_exp,
            "days_to_exp":       days_to_exp,
            "days_remaining":    days_remaining,
            "trade_expiring_soon": days_remaining is not None and days_remaining <= 2,
            "proxy_price":       round(proxy_price, 2),
            "pcr":               pcr,
            "pcr_source":        pcr_note,
            "max_pain":          mp_price,
            "max_pain_pct":      mp_pct,
            "max_pain_valid":    mp_valid,
            "implied_move_pct":  impl_pct,
            "implied_up":        impl_up,
            "implied_down":      impl_down,
            "tp_in_range":       tp_in_range,
            "skew":              skew,
            "skew_detail":       skew_detail,
            "iv_rank":           iv_rank,
            "atm_iv":            atm_iv,
            "hv_252":            round(hv_252, 1) if hv_252 else None,
            "hv_21":             round(hv_21, 1) if hv_21 else None,
            "oi_map":            oi_list,
            "oi_using_volume":   oi_using_volume,
            "call_walls":        call_walls,
            "put_walls":         put_walls,
            "signals":           signals,
            "verdict":           verdict,
            "verdict_color":     vc,
            "verdict_msg":       vm,
            "data_quality":      data_quality,
            "total_call_oi":     int(c_oi),
            "total_put_oi":      int(p_oi),
        }
    except BaseException as e:
        return {"error": str(e)}

def _recalc_metrics(trades):
    """Recalcula metrics_oos a partir de una lista de trades rich (dicts con 'pnl')."""
    pnls = np.array([t['pnl'] for t in trades if t.get('pnl') is not None])
    if len(pnls) < 4:
        return None
    wins = pnls[pnls > 0]; loss = pnls[pnls <= 0]
    n = len(pnls); wr = len(wins) / n * 100
    pf = abs(wins.sum() / loss.sum()) if loss.sum() != 0 else (99.0 if len(wins) > 0 else 0)
    dd = float(np.min(np.cumsum(pnls) - np.maximum.accumulate(np.cumsum(pnls))))
    sh = float(pnls.mean() / pnls.std() * np.sqrt(26)) if pnls.std() > 0 else 0
    p3 = float((pnls >= 3).mean() * 100)
    p5 = float((pnls >= 5).mean() * 100)
    return dict(
        n=n, wr=round(wr,3), pf=round(pf,3), sharpe=round(sh,3),
        dd=round(dd,3), p3=round(p3,3), p5=round(p5,3),
        avg_w=round(float(wins.mean()),3) if len(wins) else 0,
        avg_l=round(float(loss.mean()),3) if len(loss) else 0,
        total=round(float(pnls.sum()),3),
    )


def merge_trades_into_cache(cache, all_data):
    """Mergea trades nuevos cerrados del engine en optimal_params.json
    y recalcula metrics_oos con el historial completo acumulado.
    Clave de deduplicación: (ticker, entry_date).
    """
    updated = []
    for ticker, asset in all_data.items():
        if ticker not in cache:
            continue
        new_trades = asset.get('trades', [])
        if not new_trades:
            continue

        existing = cache[ticker].get('trades', [])
        existing_keys = {(t.get('ticker', ticker), t['entry_date']) for t in existing}

        added = 0
        for t in new_trades:
            key = (t.get('ticker', ticker), t['entry_date'])
            if key not in existing_keys:
                existing.append(t)
                existing_keys.add(key)
                added += 1

        if added > 0:
            # Ordenar por fecha de entrada
            existing.sort(key=lambda t: t['entry_date'])
            cache[ticker]['trades'] = existing

            # Recalcular métricas con historial completo
            new_metrics = _recalc_metrics(existing)
            if new_metrics:
                cache[ticker]['metrics_oos'] = new_metrics

            cache[ticker]['trades_updated_at'] = datetime.now().isoformat()
            updated.append((ticker, added, len(existing)))

    return updated


def _generate_dashboard(data, out_path):
    import json as _json
    data_js = _json.dumps(data, ensure_ascii=False)

    html = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swing Edge · Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080c14;--surface:#0d1320;--surface2:#111827;
  --border:#1a2236;--border2:#243050;
  --accent:#22d3ee;--green:#10b981;--red:#f43f5e;--yellow:#f59e0b;--blue:#6366f1;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#475569;
  --mono:'DM Mono',monospace;--sans:'DM Sans',sans-serif;--display:'Fraunces',serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;font-size:14px;-webkit-font-smoothing:antialiased}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 80% 50% at 50% -10%,rgba(34,211,238,.055) 0%,transparent 60%),radial-gradient(ellipse 40% 30% at 90% 80%,rgba(99,102,241,.035) 0%,transparent 50%);pointer-events:none;z-index:0}
.app{position:relative;z-index:1}
header{height:60px;border-bottom:1px solid var(--border);padding:0 2rem;display:flex;align-items:center;justify-content:space-between;background:rgba(8,12,20,.88);backdrop-filter:blur(20px);position:sticky;top:0;z-index:100}
.logo{display:flex;align-items:center;gap:.6rem;font-family:var(--display);font-size:1.1rem;font-weight:700;letter-spacing:-.02em;color:var(--text)}
.logo-icon{width:30px;height:30px;background:linear-gradient(135deg,var(--accent),var(--blue));border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:.8rem;box-shadow:0 0 20px rgba(34,211,238,.25);flex-shrink:0}
.logo-sub{font-family:var(--mono);font-size:.58rem;color:var(--text3);font-weight:300;letter-spacing:.1em;text-transform:uppercase;margin-top:.08rem}
.header-right{display:flex;align-items:center;gap:.75rem}
.ts{font-family:var(--mono);font-size:.63rem;color:var(--text3)}
.live-dot{width:6px;height:6px;background:var(--green);border-radius:50%;box-shadow:0 0 8px var(--green);animation:breathe 2.5s ease-in-out infinite;flex-shrink:0}
@keyframes breathe{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.5)}}
main{max-width:1680px;margin:0 auto;padding:2rem;display:flex;flex-direction:column;gap:1.5rem}
.sec-title{font-family:var(--mono);font-size:.63rem;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:.12em;display:flex;align-items:center;gap:.75rem;margin-bottom:1rem}
.sec-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border2),transparent)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;transition:border-color .2s}
.card:hover{border-color:var(--border2)}
/* SPY */
.spy-panel{background:linear-gradient(135deg,var(--surface),rgba(13,19,32,.98));border:1px solid var(--border);border-radius:14px;padding:1.25rem 1.5rem;transition:border-color .3s,box-shadow .3s}
.spy-panel.bullish{border-color:rgba(16,185,129,.25);box-shadow:0 0 40px rgba(16,185,129,.04)}
.spy-panel.bearish{border-color:rgba(244,63,94,.25);box-shadow:0 0 40px rgba(244,63,94,.04)}
.spy-head{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin-bottom:1rem}
.spy-badge{font-family:var(--mono);font-size:.6rem;font-weight:500;padding:.2rem .65rem;border-radius:20px;letter-spacing:.08em;text-transform:uppercase;display:inline-block;margin-bottom:.4rem}
.spy-badge.bull{background:rgba(16,185,129,.12);color:var(--green);border:1px solid rgba(16,185,129,.25)}
.spy-badge.bear{background:rgba(244,63,94,.12);color:var(--red);border:1px solid rgba(244,63,94,.25)}
.spy-lbl{font-size:.82rem;font-weight:600;color:var(--text)}
.spy-sub{font-size:.7rem;color:var(--text2);margin-top:.15rem}
.spy-stats{display:flex;gap:1.5rem;flex-wrap:wrap}
.spy-stat-val{font-family:var(--mono);font-size:.88rem;font-weight:500;color:var(--text)}
.spy-stat-lbl{font-size:.6rem;color:var(--text3);margin-top:.1rem}
.spy-bar-track{height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;margin-bottom:.3rem}
.spy-bar-fill{height:100%;border-radius:2px;transition:width .7s cubic-bezier(.4,0,.2,1)}
.spy-bar-lbls{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.53rem;color:var(--text3)}
/* ALERTS */
.alerts-section{display:none}.alerts-section.on{display:block}
.alerts-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(275px,1fr));gap:1rem}
.alert-card{background:linear-gradient(145deg,rgba(244,63,94,.055),rgba(244,63,94,.018));border:1px solid rgba(244,63,94,.2);border-radius:14px;padding:1.25rem;cursor:pointer;transition:all .25s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden}
.alert-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--red),transparent)}
.alert-card:hover{border-color:rgba(244,63,94,.4);transform:translateY(-3px);box-shadow:0 16px 40px rgba(244,63,94,.1)}
.alert-card.today::before{background:linear-gradient(90deg,var(--red),var(--yellow),transparent)}
.al-top{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:.4rem}
.al-ticker{font-family:var(--display);font-size:1.45rem;font-weight:700;color:var(--text);letter-spacing:-.02em}
.al-urgency{font-family:var(--mono);font-size:.57rem;padding:.18rem .52rem;border-radius:20px;text-transform:uppercase;letter-spacing:.07em;background:rgba(244,63,94,.14);color:var(--red);border:1px solid rgba(244,63,94,.28);white-space:nowrap}
.al-urgency.today-b{background:var(--red);color:#fff;border-color:var(--red);animation:pulse-b 1.5s infinite}
@keyframes pulse-b{0%,100%{box-shadow:0 0 0 0 rgba(244,63,94,.5)}50%{box-shadow:0 0 0 5px rgba(244,63,94,0)}}
.al-name{font-size:.68rem;color:var(--text3);margin-bottom:.6rem}
.al-date{font-size:.67rem;color:var(--text2);margin-bottom:.5rem}
.al-price{font-family:var(--mono);font-size:1rem;font-weight:500;color:var(--text);margin-bottom:.7rem}
.al-levels{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:.65rem}
.lvl{border-radius:9px;padding:.52rem .7rem}
.lvl.sl{background:rgba(244,63,94,.07);border:1px solid rgba(244,63,94,.16)}
.lvl.tp{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.16)}
.lvl-lbl{font-family:var(--mono);font-size:.53rem;color:var(--text3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.2rem}
.lvl-val{font-family:var(--mono);font-size:.85rem;font-weight:500}
.lvl-val.sl{color:var(--red)}.lvl-val.tp{color:var(--green)}
.lvl-pct{font-size:.6rem;opacity:.55}
.al-rr{display:flex;align-items:center;gap:.4rem;font-family:var(--mono);font-size:.68rem;padding:.32rem .6rem;border-radius:7px;margin-bottom:.6rem}
.al-rr.good{background:rgba(16,185,129,.08);color:var(--green);border:1px solid rgba(16,185,129,.18)}
.al-rr.ok{background:rgba(245,158,11,.08);color:var(--yellow);border:1px solid rgba(245,158,11,.18)}
.al-rr.bad{background:rgba(244,63,94,.08);color:var(--red);border:1px solid rgba(244,63,94,.18)}
.al-info{font-size:.63rem;color:var(--text3);margin-bottom:.6rem;line-height:1.85}
.al-chips{display:flex;flex-wrap:wrap;gap:.3rem}
.chip{font-family:var(--mono);font-size:.6rem;padding:.15rem .45rem;border-radius:5px;background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.18);color:var(--blue)}
.rr-legend{font-size:.63rem;color:var(--text3);margin-top:.8rem;padding-top:.65rem;border-top:1px solid var(--border);line-height:2}
.al-ema50{display:flex;align-items:center;gap:.38rem;font-family:var(--mono);font-size:.62rem;margin-top:.55rem;padding:.3rem .55rem;border-radius:6px}
.al-ema50.above{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.18);color:var(--green)}
.al-ema50.below{background:rgba(244,63,94,.07);border:1px solid rgba(244,63,94,.18);color:var(--red)}
.al-ema50-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.al-ema50.above .al-ema50-dot{background:var(--green);box-shadow:0 0 6px var(--green)}
.al-ema50.below .al-ema50-dot{background:var(--red);box-shadow:0 0 6px var(--red)}
/* BLOCKED */
.blocked-sec{display:none}.blocked-sec.on{display:block}
.blocked-hd{display:flex;align-items:center;gap:.55rem;margin-bottom:.55rem}
.blocked-lbl{font-size:.8rem;font-weight:600;color:var(--yellow)}
.blocked-note{font-size:.7rem;color:var(--text3);margin-bottom:.85rem;line-height:1.75;max-width:720px}
.blocked-grid{display:flex;flex-wrap:wrap;gap:.5rem}
.b-chip{background:rgba(245,158,11,.055);border:1px solid rgba(245,158,11,.18);border-radius:10px;padding:.48rem .85rem;cursor:pointer;transition:all .2s}
.b-chip:hover{background:rgba(245,158,11,.12);transform:translateY(-2px);border-color:rgba(245,158,11,.35)}
.b-tkr{font-family:var(--mono);font-size:.88rem;font-weight:500;color:var(--yellow)}
.b-name{font-size:.6rem;color:var(--text3);margin-top:.1rem}
.b-rsn{font-size:.57rem;color:var(--text3);margin-top:.15rem;opacity:.7}
/* STATS */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:.9rem}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem 1.1rem;transition:border-color .2s,transform .2s}
.stat-card:hover{border-color:var(--border2);transform:translateY(-1px)}
.stat-lbl{font-family:var(--mono);font-size:.57rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text3);margin-bottom:.4rem}
.stat-val{font-family:var(--mono);font-size:1.45rem;font-weight:500;line-height:1;margin-bottom:.22rem}
.stat-val.green{color:var(--green)}.stat-val.red{color:var(--red)}.stat-val.yellow{color:var(--yellow)}.stat-val.blue{color:var(--accent)}.stat-val.muted{color:var(--text2)}
.stat-sub{font-size:.62rem;color:var(--text3)}
/* OOS BADGE */
.oos-badge{display:inline-flex;align-items:center;gap:.4rem;font-family:var(--mono);font-size:.6rem;padding:.22rem .62rem;border-radius:20px;background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);color:var(--blue);margin-bottom:1rem}
/* TABLE */
.tbl-wrap{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.tbl{width:100%;border-collapse:collapse}
.tbl thead th{font-family:var(--mono);font-size:.58rem;text-transform:uppercase;letter-spacing:.09em;color:var(--text3);padding:.75rem 1rem;text-align:left;border-bottom:1px solid var(--border);background:rgba(255,255,255,.013)}
.tbl thead th:not(:first-child){text-align:right}
.tbl tbody tr{border-bottom:1px solid rgba(26,34,54,.6);cursor:pointer;transition:background .12s}
.tbl tbody tr:hover{background:rgba(255,255,255,.023)}
.tbl tbody tr:last-child{border-bottom:none}
.tbl tbody td{padding:.75rem 1rem;font-size:.75rem}
.tbl tbody td:not(:first-child){text-align:right;font-family:var(--mono)}
.tbl-tk{font-family:var(--mono);font-size:.88rem;font-weight:500;color:var(--text)}
.tbl-nm{font-size:.62rem;color:var(--text3);margin-top:.1rem}
.bdg{display:inline-block;font-family:var(--mono);font-size:.6rem;font-weight:500;padding:.15rem .45rem;border-radius:5px}
.bdg.g{background:rgba(16,185,129,.1);color:var(--green);border:1px solid rgba(16,185,129,.22)}
.bdg.y{background:rgba(245,158,11,.1);color:var(--yellow);border:1px solid rgba(245,158,11,.22)}
.bdg.r{background:rgba(244,63,94,.1);color:var(--red);border:1px solid rgba(244,63,94,.22)}
.bdg.live{background:var(--red);color:#fff;border:1px solid var(--red);animation:breathe 1.5s infinite}
.pos{color:var(--green)}.neg{color:var(--red)}
/* DETAIL */
.detail-panel{display:none}.detail-panel.open{display:block}
.dp-head{padding:1.25rem 1.5rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:rgba(255,255,255,.015)}
.dp-ticker{font-family:var(--display);font-size:1.9rem;font-weight:700;letter-spacing:-.03em;color:var(--text)}
.dp-name{font-size:.7rem;color:var(--text3);margin-top:.18rem}
.close-btn{width:32px;height:32px;border-radius:8px;border:1px solid var(--border);background:transparent;color:var(--text3);cursor:pointer;font-size:1rem;display:flex;align-items:center;justify-content:center;transition:all .15s}
.close-btn:hover{background:rgba(255,255,255,.06);color:var(--text);border-color:var(--border2)}
.dp-body{padding:1.5rem}
.ema50-ind{font-size:.7rem;padding:.42rem .8rem;border-radius:8px;margin-bottom:1rem;display:none}
.ema50-ind.ok{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);color:var(--green)}
.ema50-ind.bad{background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.2);color:var(--red)}
.mini-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-bottom:1.5rem}
.mini-c{background:rgba(255,255,255,.023);border:1px solid var(--border);border-radius:10px;padding:.6rem .8rem;text-align:center}
.mini-v{font-family:var(--mono);font-size:1.05rem;font-weight:500;line-height:1}
.mini-l{font-size:.57rem;color:var(--text3);margin-top:.24rem}
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;margin-bottom:1.5rem}
@media(max-width:660px){.charts-grid{grid-template-columns:1fr}}
.chart-wrap{position:relative;height:185px;background:rgba(0,0,0,.18);border-radius:10px;overflow:hidden;border:1px solid var(--border)}
canvas{display:block;width:100%;height:100%}
.dist-bars{display:flex;align-items:flex-end;gap:3px;height:52px;margin-top:.5rem}
.db-wrap{flex:1;display:flex;flex-direction:column;align-items:center}
.db-bar{width:100%;border-radius:2px 2px 0 0;min-height:2px}
.db-lbl{font-family:var(--mono);font-size:.45rem;color:var(--text3);margin-top:.2rem;white-space:nowrap}
.trades-tbl{width:100%;border-collapse:collapse;font-size:.7rem}
.trades-tbl th{text-align:left;padding:.38rem .6rem;font-family:var(--mono);font-size:.56rem;text-transform:uppercase;letter-spacing:.07em;color:var(--text3);border-bottom:1px solid var(--border)}
.trades-tbl th:not(:first-child){text-align:right}
.trades-tbl td{padding:.4rem .6rem;border-bottom:1px solid rgba(26,34,54,.5);font-family:var(--mono)}
.trades-tbl td:not(:first-child){text-align:right}
.trades-tbl tr:last-child td{border-bottom:none}
.pnl-row{display:flex;align-items:center;gap:.35rem;justify-content:flex-end}
.pnl-pill{height:3px;border-radius:2px;opacity:.5;min-width:2px}
.params-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(118px,1fr));gap:.5rem}
.param-item{background:rgba(255,255,255,.023);border:1px solid var(--border);border-radius:8px;padding:.4rem .68rem}
.param-key{font-family:var(--mono);font-size:.55rem;color:var(--text3);margin-bottom:.16rem;text-transform:uppercase;letter-spacing:.06em}
.param-val{font-family:var(--mono);font-size:.8rem;color:var(--accent);font-weight:500}
/* OPTIONS */
.opts-panel{background:linear-gradient(135deg,rgba(99,102,241,.048),rgba(34,211,238,.025));border:1px solid rgba(99,102,241,.2);border-radius:14px;padding:1.25rem 1.5rem;margin-top:1.5rem}
.opts-head{display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:.75rem;margin-bottom:1.1rem}
.opts-title{font-family:var(--mono);font-size:.63rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);font-weight:500}
.opts-proxy{font-size:.63rem;color:var(--text3);margin-top:.2rem}
.opts-verdict{font-family:var(--mono);font-size:.63rem;font-weight:500;padding:.28rem .78rem;border-radius:20px;letter-spacing:.05em;text-transform:uppercase}
.opts-verdict.green{background:rgba(16,185,129,.12);color:var(--green);border:1px solid rgba(16,185,129,.25)}
.opts-verdict.red{background:rgba(244,63,94,.12);color:var(--red);border:1px solid rgba(244,63,94,.25)}
.opts-verdict.yellow{background:rgba(245,158,11,.12);color:var(--yellow);border:1px solid rgba(245,158,11,.25)}
.opts-vmsg{font-size:.62rem;color:var(--text3);margin-top:.22rem;text-align:right}
.opts-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:.65rem;margin-bottom:1.1rem}
.opts-card{background:rgba(0,0,0,.2);border:1px solid var(--border);border-radius:10px;padding:.62rem .8rem;transition:border-color .15s}
.opts-card:hover{border-color:var(--border2)}
.opts-clbl{font-family:var(--mono);font-size:.53rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:.28rem}
.opts-cval{font-family:var(--mono);font-size:1.05rem;font-weight:500;line-height:1}
.opts-csub{font-size:.6rem;color:var(--text3);margin-top:.22rem;line-height:1.4}
.opts-sigs{display:flex;flex-direction:column;gap:.38rem;margin-bottom:1.1rem}
.sig{display:flex;align-items:flex-start;gap:.48rem;padding:.48rem .68rem;border-radius:8px;font-size:.67rem;line-height:1.5}
.sig.bullish{background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.18);color:var(--green)}
.sig.bearish{background:rgba(244,63,94,.07);border:1px solid rgba(244,63,94,.18);color:var(--red)}
.sig.neutral{background:rgba(71,85,105,.1);border:1px solid rgba(71,85,105,.22);color:var(--text2)}
.sig.warning{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.18);color:var(--yellow)}
.sig-ico{font-size:.8rem;flex-shrink:0;margin-top:.05rem}
.range-sec{margin-bottom:1.1rem}
.range-title{font-family:var(--mono);font-size:.58rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:.7rem}
.range-track{position:relative;height:30px;background:rgba(255,255,255,.04);border-radius:6px;border:1px solid var(--border);margin:0 .5rem}
.range-fill{position:absolute;top:0;height:100%;background:linear-gradient(90deg,rgba(244,63,94,.2),rgba(16,185,129,.2));border-radius:6px}
.range-mkr{position:absolute;top:-7px;bottom:-7px;width:2px;border-radius:1px}
.range-tag{position:absolute;font-family:var(--mono);font-size:.5rem;white-space:nowrap;top:-20px}
.range-note{font-size:.6rem;color:var(--text3);margin-top:.85rem;line-height:1.85}
.oi-sec{margin-bottom:1.1rem}
.oi-title{font-family:var(--mono);font-size:.58rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:.58rem}
.oi-scroll{overflow-x:auto;padding-bottom:.4rem}
.oi-chart{display:flex;align-items:flex-end;gap:4px;height:95px;min-width:380px;padding-bottom:22px;position:relative}
.oi-bwrap{display:flex;flex-direction:column;align-items:center;flex:1;height:100%;justify-content:flex-end;position:relative}
.oi-bc{border-radius:2px 2px 0 0;width:100%}
.oi-bp{width:100%}
.oi-lbl{font-family:var(--mono);font-size:.43rem;color:var(--text3);position:absolute;bottom:-20px;white-space:nowrap;transform:rotate(-40deg);transform-origin:top left}
.opts-guide{font-size:.62rem;color:var(--text3);padding-top:.72rem;border-top:1px solid rgba(255,255,255,.05);line-height:2.1}
.opts-guide strong{color:var(--text2)}
.opts-guide .key{color:var(--accent);font-family:var(--mono);font-weight:500}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeUp .35s cubic-bezier(.4,0,.2,1) both}
</style>
</head>
<body>
<div class="app">
<header>
  <div class="logo">
    <div class="logo-icon">⚡</div>
    <div>
      <div>Swing Edge</div>
      <div class="logo-sub">Trading Intelligence</div>
    </div>
  </div>
  <div class="header-right">
    <div class="live-dot"></div>
    <div class="ts" id="gt"></div>
  </div>
</header>
<main>
  <div id="spy-macro" class="spy-panel"></div>
  <div class="alerts-section" id="ab">
    <div class="sec-title">🚨 Señales de compra activas</div>
    <div class="alerts-grid" id="ac"></div>
    <div class="rr-legend" id="rr-note"></div>
  </div>
  <div class="card blocked-sec" id="mb" style="padding:1.25rem 1.5rem">
    <div class="blocked-hd"><span style="font-size:1rem">⚡</span><span class="blocked-lbl">Señales bloqueadas por filtro macro</span></div>
    <div class="blocked-note">Estos activos tienen señal técnica válida HOY pero el filtro macro (SPY bajo su EMA50) las está bloqueando. El mercado está en modo bajista — el sistema no opera en contra de la tendencia general.</div>
    <div class="blocked-grid" id="mb-grid"></div>
  </div>
  <div>
    <div class="sec-title">Resumen global del sistema</div>
    <div class="stats-grid" id="sr"></div>
  </div>
  <div>
    <div class="sec-title">Universo de activos optimizados</div>
    <div class="oos-badge">⚡ Métricas OOS — validación fuera de muestra</div>
    <div class="tbl-wrap">
      <table class="tbl">
        <thead><tr><th>Activo</th><th>Win% OOS</th><th>PF</th><th>Sharpe</th><th>Avg Win</th><th>Avg Loss</th><th>≥3%</th><th>Max DD</th><th>Total OOS</th><th>Trades</th><th>Estado</th></tr></thead>
        <tbody id="atb"></tbody>
      </table>
    </div>
  </div>
  <div class="card detail-panel" id="dp">
    <div class="dp-head">
      <div><div class="dp-ticker" id="ptk">—</div><div class="dp-name" id="ptn"></div></div>
      <button class="close-btn" onclick="closePanel()">✕</button>
    </div>
    <div class="dp-body">
      <div class="oos-badge" style="margin-bottom:1rem">⚡ Métricas OOS · Historial completo abajo</div>
      <div class="ema50-ind" id="asset-ema50"></div>
      <div class="mini-grid" id="mss"></div>
      <div class="charts-grid">
        <div><div class="sec-title" style="margin-bottom:.55rem">Precio 90d + EMAs</div><div class="chart-wrap"><canvas id="pc2"></canvas></div></div>
        <div><div class="sec-title" style="margin-bottom:.55rem">RSI · ADX</div><div class="chart-wrap"><canvas id="rc2"></canvas></div></div>
      </div>
      <div style="margin-bottom:1.5rem">
        <div class="sec-title" style="margin-bottom:.45rem">Distribución P&amp;L (historial completo)</div>
        <div class="dist-bars" id="dcc"></div>
      </div>
      <div style="margin-bottom:1.5rem">
        <div class="sec-title" style="margin-bottom:.55rem">Historial completo de trades</div>
        <div style="overflow-x:auto">
          <table class="trades-tbl">
            <thead><tr><th>Entrada</th><th>Salida</th><th>Ent$</th><th>Sal$</th><th>SL</th><th>TP</th><th>P&amp;L</th><th>Días</th><th>Razón</th></tr></thead>
            <tbody id="ttb"></tbody>
          </table>
        </div>
      </div>
      <div>
        <div class="sec-title" style="margin-bottom:.55rem">Parámetros óptimos</div>
        <div class="params-grid" id="prg"></div>
      </div>
      <div id="op-panel"></div>
    </div>
  </div>
</main>
</div>
<script>
""" + "const D=" + r"""__DATA__;

function drawChart(id,datasets,opts){
  const cv=document.getElementById(id);if(!cv)return;
  const ctx=cv.getContext('2d');
  const W=cv.offsetWidth||cv.parentElement.offsetWidth||400;
  const H=cv.offsetHeight||cv.parentElement.offsetHeight||200;
  cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);
  const pad={t:10,r:10,b:24,l:44},cW=W-pad.l-pad.r,cH=H-pad.t-pad.b;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='rgba(26,34,54,.9)';ctx.lineWidth=1;
  for(let g=0;g<=4;g++){const y=pad.t+g/4*cH;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}
  let all=datasets.flatMap(d=>d.data.filter(v=>v!=null&&!isNaN(v)));
  let mn=opts.min??Math.min(...all),mx=opts.max??Math.max(...all);
  if(mx===mn){mx+=1;mn-=1;}const rng=mx-mn;
  const xp=(i,n)=>pad.l+i/(n-1)*cW,yp=v=>pad.t+(1-(v-mn)/rng)*cH;
  (opts.refs||[]).forEach(r=>{
    ctx.strokeStyle=r.c||'rgba(255,255,255,.1)';ctx.lineWidth=1;ctx.setLineDash([3,4]);
    const y=yp(r.v);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=r.c||'rgba(255,255,255,.4)';ctx.font='7px DM Mono';ctx.fillText(r.l||r.v,W-pad.r-26,y-2);
  });
  datasets.forEach(ds=>{
    const data=ds.data,n=data.length;if(!n)return;
    ctx.strokeStyle=ds.c||'#22d3ee';ctx.lineWidth=ds.w||1.5;ctx.setLineDash(ds.d||[]);
    if(ds.fill){
      ctx.beginPath();let s=false;
      data.forEach((v,i)=>{if(v==null)return;const x=xp(i,n),y=yp(v);s?ctx.lineTo(x,y):(ctx.moveTo(x,y),s=true);});
      ctx.lineTo(xp(n-1,n),pad.t+cH);ctx.lineTo(pad.l,pad.t+cH);ctx.closePath();
      ctx.fillStyle=ds.fill;ctx.fill();
    }
    ctx.beginPath();let s=false;
    data.forEach((v,i)=>{if(v==null||isNaN(v))return;const x=xp(i,n),y=yp(v);s?ctx.lineTo(x,y):(ctx.moveTo(x,y),s=true);});
    ctx.stroke();ctx.setLineDash([]);
    if(ds.sigs)ds.sigs.forEach((sg,i)=>{
      if(sg&&data[i]!=null){
        const x=xp(i,n),y=yp(data[i]);
        ctx.fillStyle='rgba(16,185,129,.18)';ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='#10b981';ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();
      }
    });
  });
  ctx.fillStyle='rgba(71,85,105,.85)';ctx.font='7px DM Mono';
  for(let g=0;g<=4;g++){const v=mn+(1-g/4)*rng,y=pad.t+g/4*cH;ctx.fillText(v.toFixed(opts.dec??0),2,y+3);}
  if(opts.dates){const step=Math.ceil(opts.dates.length/5);opts.dates.forEach((d,i)=>{if(i%step!==0)return;ctx.fillStyle='rgba(71,85,105,.75)';ctx.fillText(d.slice(5),xp(i,opts.dates.length)-10,H-5);});}
}

if(D.generated_at){const d=new Date(D.generated_at);document.getElementById('gt').textContent='Actualizado '+d.toLocaleString('es-ES',{dateStyle:'short',timeStyle:'short'});}

const mf=D.macro_filter;
if(mf){
  const above=mf.spy_above,dist=mf.distance_pct,distAbs=Math.abs(dist).toFixed(2);
  const panel=document.getElementById('spy-macro');
  panel.classList.add(above?'bullish':'bearish');
  const bw=Math.min(Math.abs(dist)*4,100).toFixed(1);
  const blocked=mf.filter_active?`<span style="color:var(--red);font-family:var(--mono);font-size:.68rem">${mf.n_blocked} señal(es) bloqueada(s)</span>`:`<span style="color:var(--green);font-family:var(--mono);font-size:.68rem">Señales permitidas ✓</span>`;
  panel.innerHTML=`
    <div class="spy-head">
      <div>
        <span class="spy-badge ${above?'bull':'bear'}">${above?'Alcista':'Bajista'}</span>
        <div class="spy-lbl">SPY vs EMA50 — Filtro macro ${above?'desactivado':'ACTIVO'}</div>
        <div class="spy-sub">SPY cotiza un <strong style="color:${above?'var(--green)':'var(--red)'}">${above?'+':'-'}${distAbs}%</strong> ${above?'por encima':'por debajo'} de su EMA50</div>
      </div>
      <div class="spy-stats">
        <div><div class="spy-stat-val">$${mf.spy_price}</div><div class="spy-stat-lbl">Precio SPY</div></div>
        <div><div class="spy-stat-val">$${mf.spy_ema50}</div><div class="spy-stat-lbl">EMA50</div></div>
        <div style="text-align:right">${blocked}<div class="spy-stat-lbl" style="text-align:right;margin-top:.1rem">Estado señales</div></div>
      </div>
    </div>
    <div class="spy-bar-track"><div class="spy-bar-fill" style="width:${bw}%;background:${above?'var(--green)':'var(--red)'}"></div></div>
    <div class="spy-bar-lbls"><span>0%</span><span style="color:${above?'var(--green)':'var(--red)'}">${above?'+':'-'}${distAbs}% de la EMA50</span><span>±25%</span></div>`;
}

const bl=D.blocked_by_macro||[];
if(bl.length){
  document.getElementById('mb').classList.add('on');
  document.getElementById('mb-grid').innerHTML=bl.map(b=>`
    <div class="b-chip" onclick="openAsset('${b.ticker}')">
      <div class="b-tkr">${b.ticker}</div>
      <div class="b-name">${b.name||''}</div>
      ${b.motivo?`<div class="b-rsn">${b.motivo}</div>`:''}
    </div>`).join('');
}

const al=D.alerts||[];
if(al.length){
  document.getElementById('ab').classList.add('on');
  document.getElementById('ac').innerHTML=al.map(a=>{
    const rrc=a.rr_ratio>=1.5?'good':a.rr_ratio>=1.0?'ok':'bad';
    const rrl=a.rr_ratio>=1.5?'bueno ✓':a.rr_ratio>=1.0?'ajustado ⚠':'bajo ✗';
    return `<div class="alert-card fade-in ${a.days_ago===0?'today':''}" onclick="openAsset('${a.ticker}')">
      <div class="al-top">
        <div class="al-ticker">${a.ticker}</div>
        <span class="al-urgency ${a.days_ago===0?'today-b':''}">${a.urgency}</span>
      </div>
      <div class="al-name">${a.name||''}</div>
      <div class="al-date">📅 ${a.date} · <span style="color:var(--text3)">Día ${a.days_ago+1} de ${a.max_days??'?'} hábiles</span></div>
      <div class="al-price">$${a.price?.toFixed(2)}</div>
      <div class="al-levels">
        <div class="lvl sl"><div class="lvl-lbl">Stop Loss</div><div class="lvl-val sl">$${a.stop_loss?.toFixed(2)} <span class="lvl-pct">-${a.stop_pct}%</span></div></div>
        <div class="lvl tp"><div class="lvl-lbl">Take Profit</div><div class="lvl-val tp">$${a.take_profit?.toFixed(2)} <span class="lvl-pct">+${a.tp_pct}%</span></div></div>
      </div>
      <div class="al-rr ${rrc}">⚖ R/R ${a.rr_ratio??'?'}:1 <span style="opacity:.65;font-size:.63rem"> ${rrl}</span></div>
      <div class="al-info">
        <span style="color:var(--accent)">Trailing:</span> activa tras +${a.trail_act??'?'}% → stop ${a.trail_atr??'?'}×ATR del máximo<br>
        <span style="color:var(--accent)">Tiempo máx:</span> ${a.max_days??'?'} días hábiles
      </div>
      <div class="al-chips">
        ${a.rsi?`<span class="chip">RSI ${a.rsi}</span>`:''}
        ${a.adx?`<span class="chip">ADX ${a.adx}</span>`:''}
        ${a.vol_ratio?`<span class="chip">Vol ×${a.vol_ratio}</span>`:''}
        <span class="chip">Score ${a.score}/100</span>
      </div>
      ${a.ema50_ok != null ? `<div class="al-ema50 ${a.ema50_ok ? 'above' : 'below'}">
        ${a.ema50_ok
          ? `<span class="al-ema50-dot"></span>Precio por encima de EMA50${a.ema50_val ? ` (${a.ema50_val})` : ''}`
          : `<span class="al-ema50-dot"></span>Precio por debajo de EMA50${a.ema50_val ? ` (${a.ema50_val})` : ''}`
        }
      </div>` : ''}
    </div>`;
  }).join('');
  document.getElementById('rr-note').innerHTML='⚖ <strong>R/R</strong>: cuánto ganas por cada euro arriesgado · <span style="color:var(--green)">≥1.5 bueno</span> · <span style="color:var(--yellow)">1.0–1.5 ajustado</span> · <span style="color:var(--red)">&lt;1.0 desfavorable</span>';
}

const assets=Object.values(D.assets||{});
let nt=0,nw=0,tp=0,na=0,ss=[],ps=[];
assets.forEach(a=>{
  const m=a.metrics_oos||{};
  if(m.n){nt+=m.n;nw+=Math.round((m.wr||0)/100*m.n);tp+=m.total||0;if(m.sharpe)ss.push(m.sharpe);if(m.pf)ps.push(m.pf);}
  if(a.alert)na++;
});
const ash=ss.length?ss.reduce((a,b)=>a+b)/ss.length:0,apf=ps.length?ps.reduce((a,b)=>a+b)/ps.length:0,wr=nt?nw/nt*100:0;
const stats=[
  {l:'Activos',v:assets.length,c:'blue',s:'en universo'},
  {l:'Alertas activas',v:na,c:na?'red':'muted',s:'señales de compra'},
  {l:'Win Rate OOS',v:wr.toFixed(1)+'%',c:wr>=50?'green':'yellow',s:`${nw}/${nt} trades OOS`},
  {l:'Profit Factor',v:apf.toFixed(2),c:apf>=1.5?'green':'yellow',s:'promedio OOS'},
  {l:'Sharpe',v:ash.toFixed(2),c:ash>=1?'green':'yellow',s:'promedio OOS'},
  {l:'Retorno OOS',v:(tp>=0?'+':'')+tp.toFixed(1)+'%',c:tp>=0?'green':'red',s:'suma OOS'},
];
document.getElementById('sr').innerHTML=stats.map(s=>`<div class="stat-card fade-in"><div class="stat-lbl">${s.l}</div><div class="stat-val ${s.c}">${s.v}</div><div class="stat-sub">${s.s}</div></div>`).join('');

assets.sort((a,b)=>((b.metrics_oos||{}).sharpe||0)-((a.metrics_oos||{}).sharpe||0));
document.getElementById('atb').innerHTML=assets.map(a=>{
  const m=a.metrics_oos||{},ha=!!a.alert;
  const wc=m.wr>=55?'g':m.wr>=45?'y':'r',pc=m.pf>=1.5?'g':m.pf>=1?'y':'r',sc=m.sharpe>=1?'g':m.sharpe>=0.5?'y':'r',tc=m.total>=0?'pos':'neg';
  return `<tr onclick="openAsset('${a.ticker}')">
    <td><div class="tbl-tk">${a.ticker}</div><div class="tbl-nm">${a.name||''}</div></td>
    <td><span class="bdg ${wc}">${m.wr?.toFixed(1)||'—'}%</span></td>
    <td><span class="bdg ${pc}">${m.pf?.toFixed(2)||'—'}</span></td>
    <td><span class="bdg ${sc}">${m.sharpe?.toFixed(2)||'—'}</span></td>
    <td class="pos">+${m.avg_w?.toFixed(2)||'—'}%</td>
    <td class="neg">${m.avg_l?.toFixed(2)||'—'}%</td>
    <td>${m.p3?.toFixed(0)||'—'}%</td>
    <td class="neg">${m.dd?.toFixed(1)||'—'}%</td>
    <td class="${tc}">${m.total>=0?'+':''}${m.total?.toFixed(1)||'—'}%</td>
    <td>${m.n||'—'}</td>
    <td>${ha?'<span class="bdg live">🚨 SEÑAL</span>':'<span class="bdg g">OK</span>'}</td>
  </tr>`;
}).join('');

function openAsset(ticker){
  const a=D.assets[ticker];if(!a)return;
  document.getElementById('ptk').textContent=ticker;
  document.getElementById('ptn').textContent=a.name||'';
  const mf=D.macro_filter,el50=document.getElementById('asset-ema50');
  if(el50){
    if(mf&&!mf.spy_above&&a.asset_ema50_val!=null){
      const ok=a.asset_ema50_ok;
      el50.className='ema50-ind '+(ok?'ok':'bad');el50.style.display='block';
      el50.textContent=`${ok?'✅':'🔴'} EMA50 del activo: ${ok?'precio por encima':'precio por debajo'} (EMA50=${a.asset_ema50_val}) — ${ok?'señales permitidas':'bloqueadas (SPY bajista + activo bajista)'}`;
    }else{el50.style.display='none';}
  }
  const m=a.metrics_oos||{},p=a.params||{};
  const minis=[
    {v:(m.wr||0).toFixed(1)+'%',l:'Win Rate OOS',c:m.wr>=50?'var(--green)':'var(--yellow)'},
    {v:(m.pf||0).toFixed(2),l:'Profit Factor',c:m.pf>=1.5?'var(--green)':'var(--yellow)'},
    {v:(m.sharpe||0).toFixed(2),l:'Sharpe',c:m.sharpe>=1?'var(--green)':'var(--yellow)'},
    {v:'+'+(m.avg_w||0).toFixed(2)+'%',l:'Avg Win OOS',c:'var(--green)'},
    {v:(m.avg_l||0).toFixed(2)+'%',l:'Avg Loss OOS',c:'var(--red)'},
    {v:(m.dd||0).toFixed(1)+'%',l:'Max DD OOS',c:'var(--red)'},
    {v:(m.p3||0).toFixed(0)+'%',l:'≥3% trades',c:m.p3>=30?'var(--green)':'var(--yellow)'},
    {v:(m.p5||0).toFixed(0)+'%',l:'≥5% trades',c:'var(--accent)'},
    {v:(m.n||0).toString(),l:'Trades OOS',c:'var(--accent)'},
  ];
  document.getElementById('mss').innerHTML=minis.map(s=>`<div class="mini-c"><div class="mini-v" style="color:${s.c}">${s.v}</div><div class="mini-l">${s.l}</div></div>`).join('');
  const trades=a.trades||[];
  document.getElementById('ttb').innerHTML=trades.map(t=>{
    const c=t.pnl>=0?'var(--green)':'var(--red)',bw=Math.min(Math.abs(t.pnl)*4,60),stars=t.pnl>=5?' ★★':t.pnl>=3?' ★':'';
    return `<tr><td>${t.entry_date}</td><td>${t.exit_date}</td><td>$${t.entry_price}</td><td>$${t.exit_price}</td><td style="color:var(--red)">$${t.stop_loss}</td><td style="color:var(--green)">$${t.take_profit}</td><td><div class="pnl-row"><span style="color:${c}">${t.pnl>=0?'+':''}${t.pnl.toFixed(2)}%${stars}</span><div class="pnl-pill" style="width:${bw}px;background:${c}"></div></div></td><td>${t.days}d</td><td style="font-size:.62rem;color:var(--text3)">${t.reason}</td></tr>`;
  }).join('');
  const kp=['rsi_p','rsi_lo','rsi_hi','ema_f','ema_s','ema_t','macd_f','macd_s','adx_min','score_min','tp_pct','atr_stop','max_days','vol_min','trail_act','trail_atr'];
  document.getElementById('prg').innerHTML=kp.filter(k=>p[k]!==undefined).map(k=>`<div class="param-item"><div class="param-key">${k}</div><div class="param-val">${p[k]}</div></div>`).join('');
  if(trades.length){
    const pnls=trades.map(t=>t.pnl);
    const bins=[[-99,-10],[-10,-5],[-5,-3],[-3,0],[0,3],[3,5],[5,10],[10,99]],lbls=['<-10','-10→-5','-5→-3','-3→0','0→3','3→5','5→10','>10'];
    const cnts=bins.map(([lo,hi])=>pnls.filter(p=>p>lo&&p<=hi).length),mx=Math.max(...cnts,1);
    document.getElementById('dcc').innerHTML=cnts.map((c,i)=>{
      const h=Math.round(c/mx*48),lo=bins[i][0],col=lo>=3?'var(--green)':lo>=0?'rgba(16,185,129,.4)':lo>=-5?'rgba(244,63,94,.4)':'var(--red)';
      return `<div class="db-wrap"><div class="db-bar" style="height:${h}px;background:${col}"></div><div class="db-lbl">${lbls[i]}</div></div>`;
    }).join('');
  }
  document.getElementById('dp').classList.add('open');
  const alertData=D.alerts?D.alerts.find(x=>x.ticker===ticker):null;
  const ep=alertData?alertData.price:null,tp2=alertData?alertData.take_profit:null,sl2=alertData?alertData.stop_loss:null;
  renderOptions(a.options||null,ep,tp2,sl2);
  setTimeout(()=>{
    document.getElementById('dp').scrollIntoView({behavior:'smooth',block:'start'});
    const h=a.price_history||[];
    if(h.length){
      drawChart('pc2',[
        {data:h.map(x=>x.close),c:'#64748b',w:1.5,fill:'rgba(34,211,238,.04)',sigs:h.map(x=>x.signal===1)},
        {data:h.map(x=>x.ema_f),c:'#f59e0b',w:1.3},
        {data:h.map(x=>x.ema_s),c:'#22d3ee',w:1.5},
        {data:h.map(x=>x.ema_t),c:'#a78bfa',w:1,d:[4,4]},
      ],{dec:1,dates:h.map(x=>x.date)});
      drawChart('rc2',[
        {data:h.map(x=>x.rsi),c:'#f43f5e',w:1.5},
        {data:h.map(x=>x.adx),c:'#6366f1',w:1.3,d:[3,3]},
      ],{min:0,max:100,dec:0,dates:h.map(x=>x.date),refs:[
        {v:70,c:'rgba(244,63,94,.4)',l:'70'},{v:50,c:'rgba(255,255,255,.1)',l:'50'},
        {v:30,c:'rgba(16,185,129,.35)',l:'30'},{v:25,c:'rgba(99,102,241,.35)',l:'ADX25'},
      ]});
    }
  },60);
}
function closePanel(){document.getElementById('dp').classList.remove('open');}
function fmtOI(n){if(n==null||isNaN(n))return'—';if(n===0)return'0';if(n>=1000000)return(n/1000000).toFixed(1).replace(/\.0$/,'')+'M';if(n>=1000)return(n/1000).toFixed(1).replace(/\.0$/,'')+'K';return n.toString();}
function renderOptions(opt,entry_price,take_profit,stop_loss){
  const el=document.getElementById('op-panel');
  if(!opt||opt.error){el.innerHTML=opt?`<div class="opts-panel" style="color:var(--text3);font-size:.72rem">📊 Opciones no disponibles: ${opt.error}</div>`:'';return;}
  const vc=opt.verdict_color||'yellow',sigIcons={bullish:'📈',bearish:'📉',neutral:'⚖',warning:'⚠'};
  let rangeHtml='';
  if(opt.implied_down&&opt.implied_up&&entry_price){
    const lo=Math.min(opt.implied_down,stop_loss,entry_price)*.98,hi=Math.max(opt.implied_up,take_profit,entry_price)*1.02,rng=hi-lo;
    const pct=v=>((v-lo)/rng*100).toFixed(1);
    const mkr=(v,col,label,lc)=>`<div class="range-mkr" style="left:${pct(v)}%;background:${col}"><div class="range-tag" style="left:2px;color:${lc||col}">${label}<br><span style="opacity:.7">${v.toFixed(2)}</span></div></div>`;
    rangeHtml=`<div class="range-sec"><div class="range-title">📐 Rango implícito vs tus niveles</div><div style="padding-top:24px"><div class="range-track"><div class="range-fill" style="left:${pct(opt.implied_down)}%;width:${(pct(opt.implied_up)-pct(opt.implied_down)).toFixed(1)}%"></div>${mkr(opt.implied_down,'#f43f5e','↓ Impl','#f43f5e')}${mkr(stop_loss,'#fb7185','SL','#fb7185')}${mkr(entry_price,'#e2e8f0','ENTRADA','#e2e8f0')}${mkr(take_profit,opt.tp_in_range?'#10b981':'#f59e0b','TP',opt.tp_in_range?'#10b981':'#f59e0b')}${mkr(opt.implied_up,'#10b981','↑ Impl','#10b981')}</div></div><div class="range-note">Rango: <span style="color:var(--text)">$${opt.implied_down?.toFixed(2)} → $${opt.implied_up?.toFixed(2)}</span> (±${opt.implied_move_pct}% para ${opt.days_remaining??opt.days_to_exp}d · IV ann. ${opt.atm_iv??'—'}% · ref. ${opt.expiration}) &nbsp;·&nbsp; Tu TP: <span style="color:${opt.tp_in_range?'var(--green)':'var(--yellow)'}">${opt.tp_in_range?'✓ dentro':'⚠ fuera'} del rango</span></div></div>`;
  }
  let oiHtml='';
  if(opt.oi_map&&opt.oi_map.length){
    const maxOI=Math.max(...opt.oi_map.map(x=>x.total_oi),1),atm=opt.proxy_price;
    oiHtml=`<div class="oi-sec"><div class="oi-title">📊 Open Interest por strike <span style="margin-left:.5rem;font-size:.53rem"><span style="color:rgba(99,102,241,.9)">■</span> Calls &nbsp;<span style="color:rgba(244,63,94,.9)">■</span> Puts</span></div><div class="oi-scroll"><div class="oi-chart">${opt.oi_map.map(bar=>{const hC=Math.round(bar.call_oi/maxOI*68),hP=Math.round(bar.put_oi/maxOI*68),isAtm=Math.abs(bar.strike_proxy-atm)/atm<0.015;return`<div class="oi-bwrap" style="${isAtm?'outline:1px solid rgba(34,211,238,.4);border-radius:3px':''}"><div style="display:flex;flex-direction:column;align-items:center;width:100%;justify-content:flex-end;height:68px;gap:0"><div class="oi-bc" style="height:${hC}px;background:rgba(99,102,241,.75)"></div><div class="oi-bp" style="height:${hP}px;background:rgba(244,63,94,.7)"></div></div><div class="oi-lbl">${bar.strike_asset?.toFixed(1)||bar.strike_proxy}</div></div>`;}).join('')}</div></div><div style="font-size:.58rem;color:var(--text3);margin-top:.38rem">Calls altas = resistencia · Puts altas = soporte · ATM marcado en cyan</div></div>`;
  }
  const pcrC=opt.pcr==null?'var(--text2)':opt.pcr>1.5?'var(--red)':opt.pcr>1.2?'#fb923c':opt.pcr<0.6?'var(--green)':opt.pcr<0.8?'#6ee7b7':'var(--yellow)';
  const skC=opt.skew==null?'var(--text2)':opt.skew>25?'var(--red)':opt.skew>10?'#fb923c':opt.skew<-10?'var(--green)':'var(--yellow)';
  const mpC=opt.max_pain_pct>3?'var(--green)':opt.max_pain_pct<-3?'var(--red)':'var(--yellow)';
  const ivC=opt.iv_rank>70?'var(--red)':opt.iv_rank<30?'var(--green)':'var(--yellow)';
  const dqC=opt.data_quality==='alta'?'var(--green)':opt.data_quality==='media'?'var(--yellow)':'var(--red)';
  const cards=[
    {l:'Put/Call Ratio',v:opt.pcr??'—',c:pcrC,s:opt.pcr==null?(opt.pcr_source||'sin datos'):opt.pcr>1.5?'Muy bajista · contrarian posible':opt.pcr>1.2?'Bajista moderado':opt.pcr<0.6?'Alcista fuerte':opt.pcr<0.8?'Leve sesgo alcista':'Neutro (0.8–1.2)'},
    {l:'Implied Move',v:opt.implied_move_pct!=null?`±${opt.implied_move_pct}%`:'—',c:'var(--accent)',s:opt.implied_move_pct!=null?`Para ${opt.days_remaining??opt.days_to_exp}d · IV ann. ${opt.atm_iv??'—'}% · ${opt.expiration}`:'No disponible'},
    {l:'Max Pain',v:opt.max_pain_valid?`${opt.max_pain}`:'—',c:mpC,s:opt.max_pain_valid?`${opt.max_pain_pct>0?'+':''}${opt.max_pain_pct}% vs precio`:'OI insuf.'},
    {l:'Skew Puts/Calls',v:opt.skew!=null?`${opt.skew>0?'+':''}${opt.skew}%`:'—',c:skC,s:opt.skew!=null?`Put ${opt.skew_detail?.put_strike??'?'} vs Call ${opt.skew_detail?.call_strike??'?'}`:(opt.skew_detail?.note||'Sin datos')},
    {l:'IV Rank',v:opt.iv_rank!=null?`${opt.iv_rank}%`:'—',c:ivC,s:opt.iv_rank!=null?(opt.iv_rank>70?'Opciones CARAS':opt.iv_rank<30?'Opciones BARATAS':'IV normal'):'Sin histórico'},
    {l:'IV ATM',v:opt.atm_iv!=null?`${opt.atm_iv}%`:'—',c:'var(--accent)',s:opt.hv_252!=null?`HV 252d: ${opt.hv_252}% · HV 21d: ${opt.hv_21??'?'}%`:'Sin HV'},
    {l:'Call OI',v:opt.total_call_oi!=null?fmtOI(opt.total_call_oi):'—',c:'rgba(99,102,241,.9)',s:opt.oi_using_volume?'⚠ Usando volumen':'Open Interest calls'},
    {l:'Put OI',v:opt.total_put_oi!=null?fmtOI(opt.total_put_oi):'—',c:'rgba(244,63,94,.9)',s:opt.oi_using_volume?'⚠ Usando volumen':'Open Interest puts'},
    {l:'Calidad datos',v:opt.data_quality?.toUpperCase()??'—',c:dqC,s:`Venc: ${opt.expiration} · ${opt.days_to_exp}d`},
  ];
  el.innerHTML=`<div class="opts-panel">
    <div class="opts-head">
      <div>
        <div class="opts-title">📊 Microestructura de opciones</div>
        <div class="opts-proxy">${opt.is_proxy?`Proxy: ${opt.options_ticker}`:`Opciones directas: ${opt.options_ticker}`} · Próx. venc: ${opt.expiration}</div>
      </div>
      <div style="text-align:right">
        <div class="opts-verdict ${vc}">${opt.verdict}</div>
        <div class="opts-vmsg">${opt.verdict_msg}</div>
      </div>
    </div>
    <div class="opts-cards">${cards.map(c=>`<div class="opts-card"><div class="opts-clbl">${c.l}</div><div class="opts-cval" style="color:${c.c}">${c.v}</div><div class="opts-csub">${c.s}</div></div>`).join('')}</div>
    ${rangeHtml}${oiHtml}
    <div class="opts-sigs">${(opt.signals||[]).map(s=>`<div class="sig ${s.type}"><span class="sig-ico">${sigIcons[s.type]||'·'}</span><span>${s.msg}</span></div>`).join('')}</div>
    <div class="opts-guide"><strong>📖 Guía:</strong><br>
      <span class="key">PCR</span> &gt;1.5 = miedo/cobertura bajista (contrarian alcista posible) · &lt;0.6 = confianza alcista · 0.8–1.2 = neutro.<br>
      <span class="key">Implied Move</span>: rango esperado hasta vencimiento. Tu TP debería estar dentro.<br>
      <span class="key">Max Pain</span>: precio donde más opciones expiran sin valor — imán antes del vencimiento.<br>
      <span class="key">Skew</span>: diferencia IV puts/calls OTM. Positivo = miedo a caída. &gt;25% = pánico. Negativo = anticipación alcista.<br>
      <span class="key">IV Rank</span>: &gt;70% opciones caras (vender). &lt;30% baratas (comprar).<br>
      <span style="color:var(--yellow)">⚠ Calidad BAJA</span>: proxy sin liquidez suficiente — usa señales técnicas como fuente principal.
    </div>
  </div>`;
}
</script>
</body>
</html>"""


    html = html.replace('__DATA__', data_js)
    out_path.write_text(html, encoding='utf-8')
    print(f"  Dashboard: {out_path}")


def main():
    TICKER_SOLO  = None
    SOLO_ALERTAS = False

    if not CACHE_FILE.exists():
        print(f"{R}Error: optimal_params.json no encontrado.{RST}")
        return

    cache = json.loads(CACHE_FILE.read_text())
    print(f"\n{B}{'═'*65}{RST}")
    print(f"{BOLD}{C}  ⚡  SWING EDGE ENGINE  v2{RST}")
    print(f"{DIM}  {len(cache)} activos · métricas OOS · historial completo{RST}")
    print(f"{B}{'═'*65}{RST}\n")

    print("  Descargando SPY (1y para filtro macro)...")
    spy = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns=spy.columns.get_level_values(0)
    spy_c = spy['Close'].squeeze()
    spy_e = spy_c.ewm(span=50, adjust=False).mean()
    spy_above = bool(spy_c.iloc[-1] > spy_e.iloc[-1])
    print(f"  SPY={spy_c.iloc[-1]:.2f} EMA50={spy_e.iloc[-1]:.2f} → {'✅ ALCISTA' if spy_above else '🔴 BAJISTA'} (filtro macro {'OFF' if spy_above else 'ACTIVO'})")

    tickers = [TICKER_SOLO] if TICKER_SOLO else list(cache.keys())
    alerts   = []
    blocked  = []   # señales bloqueadas por filtro macro
    all_data = {}

    for ticker in tickers:
        if ticker not in cache:
            print(f"  {Y}⚠ {ticker} no en caché{RST}"); continue

        entry  = cache[ticker]
        p      = entry['params']
        name   = UNIVERSE.get(ticker, ticker)
        # ── CAMBIO CLAVE: usar métricas OOS del optimizer ──────
        m_oos  = entry.get('metrics_oos', {})

        print(f"  ↓ {ticker}...", end=" ", flush=True)
        try:
            df_raw = yf.download(ticker, period=PERIOD, auto_adjust=True, progress=False)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)
            if len(df_raw) < 60:
                print("❌ pocos datos"); continue
            print(f"✓ {len(df_raw)}d")
        except Exception as e:
            print(f"❌ {e}"); continue

        # ── Filtro macro combinado ────────────────────────────────
        # Si SPY alcista → macro = True siempre (todas las señales pasan)
        # Si SPY bajista → macro = True solo si activo está por encima de su EMA50
        #   (el activo va contra la corriente bajista del mercado)
        if not spy_above:
            asset_close = df_raw['Close'].squeeze()
            asset_ema50 = asset_close.ewm(span=50, adjust=False).mean()
            asset_above_series = (asset_close > asset_ema50)
            # macro[i] = True si el activo está por encima de su EMA50 en esa barra
            # ignoramos completamente spy_filter — ya sabemos que SPY es bajista
            macro = asset_above_series.reindex(df_raw.index, method='ffill').fillna(False).values
            asset_ema50_ok  = bool(asset_close.iloc[-1] > asset_ema50.iloc[-1])
            asset_ema50_val = round(float(asset_ema50.iloc[-1]), 2)
        else:
            # SPY alcista: macro = True siempre
            macro = np.ones(len(df_raw), dtype=bool)
            asset_ema50_ok  = True
            asset_ema50_val = None

        use_invert = entry.get('invert', ticker in INVERSE_TICKERS or ticker in VIX_TICKERS)
        ind    = calc_ind(df_raw, p, use_invert)
        sigs   = get_signals(ind, p, macro, use_invert)

        # ── Diagnóstico: señales sin filtro macro vs con filtro ────
        sigs_raw  = get_signals(ind, p, macro=None, invert_macro=False)
        n_raw     = int(sigs_raw.sum())
        n_final   = int(sigs.sum())
        n_blocked = n_raw - n_final

        scores = np.array([score_bar(ind, i, p) if i>=35 else 0
                           for i in range(len(ind['c']))])

        # Historial completo de trades + trade actualmente abierto
        # Si el backtest con macro no detecta open_trade, intentar con sigs_raw
        # Evita que un trade abierto desaparezca por reajuste de EMA50 al añadir barras
        trades, open_trade = build_rich_trades(ind, sigs, p, ticker)
        if open_trade is None:
            _, open_trade_raw = build_rich_trades(ind, sigs_raw, p, ticker)
            if open_trade_raw:
                current_real = float(ind['close'][-1])
                if current_real > open_trade_raw['stop_loss']:
                    open_trade = open_trade_raw

        # ── Terminal: mostrar métricas OOS ─────────────────────
        if not SOLO_ALERTAS and m_oos:
            print_summary(ticker, name, m_oos, p)
            print_trades(trades, ticker)

        alert = check_alert(ind, sigs, p, ticker, name, trades, open_trade)

        # Enriquecer alerta con estado EMA50 actual del activo
        if alert:
            alert['ema50_ok']  = asset_ema50_ok
            alert['ema50_val'] = asset_ema50_val

        # ── Bloqueadas por macro: solo si NO hay alerta activa ──
        # Un activo con trade abierto nunca puede estar en "bloqueadas" —
        # el filtro macro solo aplica en la entrada, no cancela trades en curso
        recent_blocked = False
        if not alert:
            n_total = len(ind['c'])
            if not spy_above and asset_ema50_ok:
                recent_blocked = False  # activo por encima de EMA50 → señales permitidas
            else:
                for i in range(n_total-1, max(n_total-4, 35), -1):
                    if sigs_raw[i]==1 and sigs[i]==0:
                        recent_blocked = True; break
        if recent_blocked:
            if not spy_above and not asset_ema50_ok:
                motivo = f"SPY bajista + {ticker} bajo EMA50"
            elif not spy_above:
                motivo = "SPY bajo EMA50"
            else:
                motivo = "filtro macro"
            print(f"  {Y}⚡ Señal reciente bloqueada por macro ({motivo}){RST}")
            blocked.append({"ticker": ticker, "name": name, "motivo": motivo})
        elif n_blocked > 0:
            print(f"  {DIM}Señales históricas: {n_raw} brutas → {n_final} tras macro ({n_blocked} bloqueadas){RST}")
        opt_data = None
        if alert:
            alerts.append(alert)
            print_alert(alert)
            print(f"  📊 Obteniendo opciones...", end=" ", flush=True)
            try:
                days_remaining = max(1, p['max_days'] - alert['days_ago'])
                opt_data = fetch_options_data(ticker, alert['price'], alert['take_profit'], alert['stop_loss'],
                                              days_remaining=days_remaining)
                if opt_data and not opt_data.get('error'):
                    print(f"✓ PCR={opt_data.get('pcr','—')} ImplMove=±{opt_data.get('implied_move_pct','—')}% Veredicto={opt_data.get('verdict','—')}")
                else:
                    print(f"⚠ {opt_data.get('error','sin datos') if opt_data else 'sin datos'}")
            except Exception as e_opt:
                print(f"⚠ Error opciones: {e_opt}")
                opt_data = None

        price_hist = build_price_history(ind, sigs, scores)
        all_data[ticker] = {
            "ticker":        ticker,
            "name":          name,
            "params":        p,
            "metrics_oos":   m_oos,
            "trades":        trades,
            "price_history": price_hist,
            "alert":         alert,
            "options":       opt_data,
            "optimized_at":  entry.get('optimized_at','')[:10],
            "asset_ema50_ok":  asset_ema50_ok,
            "asset_ema50_val": asset_ema50_val,
        }

    if alerts:
        print(f"\n{M}{'═'*65}{RST}")
        print(f"{BOLD}{M}  🚨 {len(alerts)} ALERTA(S) ACTIVA(S){RST}")
        print(f"{M}{'═'*65}{RST}")
        for a in sorted(alerts, key=lambda x: x['days_ago']):
            col = M if a['days_ago']==0 else Y
            print(f"  {col}{a['urgency']:<8}{RST}  {BOLD}{a['ticker']:<8}{RST}"
                  f"  ${a['price']:.2f}"
                  f"  SL:{R}${a['stop_loss']:.2f}{RST}"
                  f"  TP:{G}${a['take_profit']:.2f}{RST}"
                  f"  R/R:{a.get('rr_ratio','?')}  Score:{a['score']}")
    else:
        print(f"\n  {Y}Sin alertas activas hoy.{RST}")

    dashboard_data = {
        "generated_at":    datetime.now().isoformat(),
        "alerts":          alerts,
        "blocked_by_macro": blocked,
        "assets":          all_data,
        "macro_filter": {
            "spy_price":    round(float(spy_c.iloc[-1]), 2),
            "spy_ema50":    round(float(spy_e.iloc[-1]), 2),
            "spy_above":    spy_above,
            "distance_pct": round((float(spy_c.iloc[-1]) / float(spy_e.iloc[-1]) - 1) * 100, 2),
            "filter_active": not spy_above,
            "n_blocked":    len(blocked),
        },
    }

    def _sanitize(obj):
        """Convierte recursivamente tipos numpy a tipos Python nativos."""
        if isinstance(obj, dict):   return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [_sanitize(v) for v in obj]
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):  return [_sanitize(v) for v in obj.tolist()]
        if isinstance(obj, float) and np.isnan(obj): return None
        return obj

    # ── Merge trades nuevos en optimal_params.json ──────────────
    # Acumula el historial día a día y recalcula métricas OOS
    updated = merge_trades_into_cache(cache, all_data)
    if updated:
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
        print(f"\n  {G}→ optimal_params.json actualizado con trades nuevos:{RST}")
        for ticker, added, total in updated:
            print(f"     {BOLD}{ticker}{RST}: +{added} trade(s) nuevo(s) · historial total: {total} trades")
    else:
        print(f"\n  {DIM}→ optimal_params.json sin cambios (sin trades nuevos cerrados){RST}")

    clean_data = _sanitize(dashboard_data)
    OUTPUT_FILE.write_text(json.dumps(clean_data, ensure_ascii=False, indent=2))
    _generate_dashboard(clean_data, BASE_DIR / "dashboard.html")

    print(f"\n  {G}→ dashboard_data.json generado{RST}")
    print(f"  {G}→ dashboard.html generado (abre directamente en el navegador){RST}\n")

if __name__ == "__main__":
    main()

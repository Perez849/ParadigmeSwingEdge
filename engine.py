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
    out['close']  = out['c']
    out['high']   = h.values.astype(float)
    out['low']    = l.values.astype(float)
    out['open']   = df['Open'].squeeze().values.astype(float) if 'Open' in df.columns else out['c']
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
    c=ind['c']; atr=ind['atr']; idx=ind['index']; n=len(c)
    trades=[]; in_t=False; ep=sl=tp=pk=0.0; entry_i=0

    for i in range(n):
        price=c[i]
        if np.isnan(price): continue
        a = atr[i] if not np.isnan(atr[i]) else price*0.02

        if not in_t:
            if sigs[i]==1:
                in_t=True; ep=price; pk=price
                sl=price-a*p['atr_stop']
                tp=price*(1+p['tp_pct']/100)
                entry_i=i
        else:
            pk=max(pk,price); held=i-entry_i
            pnl=(price-ep)/ep*100; reason=None
            if price<=sl:    reason="Stop Loss"
            elif price>=tp:  reason="Take Profit ✅"
            elif pnl>p['trail_act']:
                if price<pk-a*p['trail_atr']: reason="Trailing Stop"
            if reason is None and held>=p['max_days']:
                reason=f"Tiempo ({held}d)"
            if reason:
                entry_sc = int(score_bar(ind, entry_i, p))
                trades.append({
                    "ticker":      ticker,
                    "entry_date":  str(idx[entry_i])[:10],
                    "exit_date":   str(idx[i])[:10],
                    "entry_price": round(ep,2),
                    "exit_price":  round(price,2),
                    "stop_loss":   round(ep - ind['atr'][entry_i]*p['atr_stop'],2),
                    "take_profit": round(tp,2),
                    "pnl":         round(pnl,2),
                    "peak_pnl":    round((pk-ep)/ep*100,2),
                    "days":        held,
                    "reason":      reason,
                    "score":       entry_sc,
                    "entry_rsi":   round(float(ind['rsi'][entry_i]),1) if not np.isnan(ind['rsi'][entry_i]) else None,
                    "entry_adx":   round(float(ind['adx'][entry_i]),1) if not np.isnan(ind['adx'][entry_i]) else None,
                })
                in_t=False
    return trades

def check_alert(ind, sigs, p, ticker, name, trades=None):
    closed_dates = set()
    if trades:
        today_str = str(ind['index'][-1])[:10]
        for t in trades:
            entry = t['entry_date']
            exit_  = t['exit_date']
            entry_ts = pd.Timestamp(entry)
            today_ts = pd.Timestamp(today_str)
            if (today_ts - entry_ts).days <= 4 and exit_ < today_str:
                closed_dates.add(entry)

    n = len(ind['c'])
    for i in range(n-1, max(n-4, 35), -1):
        if sigs[i] != 1: continue
        price    = float(ind['c'][i])
        a        = float(ind['atr'][i]) if not np.isnan(ind['atr'][i]) else price*0.02
        today    = ind['index'][-1]
        date     = ind['index'][i]
        date_str = str(date)[:10]
        days_ago = (pd.Timestamp(today) - pd.Timestamp(date)).days
        atr_pct  = a/price*100

        if date_str in closed_dates:
            return None

        return {
            "ticker":      ticker,
            "name":        name,
            "date":        date_str,
            "urgency":     "HOY" if days_ago==0 else f"HACE {days_ago}d",
            "days_ago":    days_ago,
            "price":       round(price,2),
            "stop_loss":   round(price - a*p['atr_stop'],2),
            "take_profit": round(price*(1+p['tp_pct']/100),2),
            "stop_pct":    round(p['atr_stop']*atr_pct,2),
            "tp_pct":      p['tp_pct'],
            "rr_ratio":    round(p['tp_pct']/(p['atr_stop']*atr_pct),2) if atr_pct>0 else None,
            "score":       int(score_bar(ind, i, p)),
            "rsi":         round(float(ind['rsi'][i]),1) if not np.isnan(ind['rsi'][i]) else None,
            "adx":         round(float(ind['adx'][i]),1) if not np.isnan(ind['adx'][i]) else None,
            "vol_ratio":   round(float(ind['vol_r'][i]),2) if not np.isnan(ind['vol_r'][i]) else None,
            "max_days":    p['max_days'],
            "trail_act":   p['trail_act'],
            "trail_atr":   p['trail_atr'],
            "trail_sl":    round(price - a * p['trail_atr'], 2),
        }
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
            "low":    gv('low'),   "close": gv('close'),
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

def _generate_dashboard(data, out_path):
    import json as _json
    data_js = _json.dumps(data, ensure_ascii=False)

    html = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swing Edge · Trading Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#060810;--card:#111720;--border:#1e2840;--accent:#00e5a0;--accent2:#4d9fff;--warn:#ffb300;--red:#ff4757;--green:#00e5a0;--text:#e8ecf4;--muted:#5a6a8a;--yellow:#ffb300}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'JetBrains Mono',monospace;min-height:100vh}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,229,160,.02) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,160,.02) 1px,transparent 1px);background-size:44px 44px;pointer-events:none;z-index:0}
.app{position:relative;z-index:1}
header{border-bottom:1px solid var(--border);padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:64px;background:rgba(6,8,16,.9);backdrop-filter:blur(12px);position:sticky;top:0;z-index:100}
.logo{font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem;display:flex;align-items:center;gap:.5rem}
.dot{width:8px;height:8px;background:var(--accent);border-radius:50%;box-shadow:0 0 10px var(--accent);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.5)}}
.hm{font-size:.68rem;color:var(--muted);text-align:right}
main{max-width:1600px;margin:0 auto;padding:2rem;display:grid;gap:1.5rem}
.ab{background:linear-gradient(135deg,rgba(255,71,87,.07),rgba(255,179,0,.04));border:1px solid rgba(255,71,87,.25);border-radius:12px;padding:1.2rem 1.5rem;display:none}
.ab.on{display:block}
.ab-h{font-family:'Syne',sans-serif;font-weight:700;font-size:.78rem;color:var(--warn);text-transform:uppercase;letter-spacing:.1em;margin-bottom:1rem}
.ac{display:grid;grid-template-columns:repeat(auto-fill,minmax(265px,1fr));gap:1rem}
.acard{background:rgba(255,71,87,.06);border:1px solid rgba(255,71,87,.18);border-radius:10px;padding:1rem 1.2rem;cursor:pointer;transition:all .2s}
.acard:hover{background:rgba(255,71,87,.12);transform:translateY(-2px)}
.acard.today{border-color:rgba(255,71,87,.5);background:rgba(255,71,87,.1)}
.at{font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem}
.au{font-size:.6rem;font-weight:600;padding:.18rem .45rem;border-radius:4px;background:rgba(255,71,87,.18);color:var(--red);letter-spacing:.08em}
.au.today{background:var(--red);color:#fff}
.ar{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.35rem}
.an{color:var(--muted);font-size:.67rem;margin-bottom:.55rem}
.stgrid{display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.45rem}
.slb{background:rgba(255,71,87,.1);border:1px solid rgba(255,71,87,.2);color:var(--red);border-radius:6px;padding:.38rem .55rem;font-size:.68rem}
.tpb{background:rgba(0,229,160,.07);border:1px solid rgba(0,229,160,.18);color:var(--green);border-radius:6px;padding:.38rem .55rem;font-size:.68rem}
.bl{font-size:.57rem;color:var(--muted);margin-bottom:.12rem}
.rrb{margin-top:.45rem;font-size:.67rem;color:var(--accent2)}
.chips{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.55rem}
.chip{font-size:.58rem;padding:.12rem .38rem;border-radius:4px;background:rgba(77,159,255,.08);border:1px solid rgba(77,159,255,.18);color:var(--accent2)}
.sr{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:1rem}
.sc{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.9rem 1.1rem}
.sl{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.38rem}
.sv{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700}
.sv.g{color:var(--green)}.sv.r{color:var(--red)}.sv.y{color:var(--yellow)}.sv.b{color:var(--accent2)}.sv.m{color:var(--muted)}
.ss{font-size:.6rem;color:var(--muted);margin-top:.18rem}
.st{font-family:'Syne',sans-serif;font-weight:700;font-size:.74rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}
.st::after{content:'';flex:1;height:1px;background:var(--border)}
.oos-note{font-size:.6rem;color:var(--accent2);margin-bottom:1rem;padding:.4rem .7rem;background:rgba(77,159,255,.06);border:1px solid rgba(77,159,255,.15);border-radius:6px;display:inline-block}
.tw{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}
table{width:100%;border-collapse:collapse}
thead th{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;padding:.7rem .9rem;text-align:left;border-bottom:1px solid var(--border);background:rgba(255,255,255,.018)}
thead th:not(:first-child){text-align:right}
tbody tr{border-bottom:1px solid rgba(30,40,64,.4);cursor:pointer;transition:background .12s}
tbody tr:hover{background:rgba(255,255,255,.025)}
tbody tr:last-child{border-bottom:none}
tbody td{padding:.72rem .9rem;font-size:.73rem}
tbody td:not(:first-child){text-align:right}
.tt{font-family:'Syne',sans-serif;font-weight:700;font-size:.86rem}
.tn{font-size:.6rem;color:var(--muted);margin-top:.1rem}
.bdg{display:inline-block;font-size:.58rem;padding:.13rem .38rem;border-radius:4px;font-weight:500}
.bg{background:rgba(0,229,160,.1);color:var(--green);border:1px solid rgba(0,229,160,.18)}
.by{background:rgba(255,179,0,.1);color:var(--yellow);border:1px solid rgba(255,179,0,.18)}
.br{background:rgba(255,71,87,.1);color:var(--red);border:1px solid rgba(255,71,87,.18)}
.ba{background:var(--red);color:#fff;animation:pulse 1.5s infinite}
.pos{color:var(--green)}.neg{color:var(--red)}
.dp{display:none;background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.dp.open{display:block}
.ph{padding:1.1rem 1.4rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:rgba(255,255,255,.018)}
.ptk{font-family:'Syne',sans-serif;font-weight:800;font-size:1.45rem;letter-spacing:-.03em}
.pc{width:30px;height:30px;border:1px solid var(--border);border-radius:6px;background:none;color:var(--muted);cursor:pointer;font-size:.9rem;display:flex;align-items:center;justify-content:center;transition:all .15s}
.pc:hover{background:rgba(255,255,255,.04);color:var(--text)}
.pb{padding:1.4rem}
.pg{display:grid;grid-template-columns:1fr 1fr;gap:1.4rem;margin-bottom:1.4rem}
@media(max-width:700px){.pg{grid-template-columns:1fr}}
.ms{display:grid;grid-template-columns:repeat(3,1fr);gap:.55rem;margin-bottom:1.4rem}
.ms-c{background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:8px;padding:.5rem .75rem;text-align:center}
.mv{font-family:'Syne',sans-serif;font-weight:700;font-size:1rem}
.ml{font-size:.57rem;color:var(--muted);margin-top:.12rem}
.cw{position:relative;height:195px;background:rgba(0,0,0,.2);border-radius:8px;overflow:hidden;border:1px solid var(--border)}
canvas{display:block;width:100%;height:100%}
.tt2{width:100%;border-collapse:collapse;font-size:.68rem}
.tt2 th{text-align:left;padding:.38rem .55rem;color:var(--muted);font-size:.58rem;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)}
.tt2 th:not(:first-child){text-align:right}
.tt2 td{padding:.4rem .55rem;border-bottom:1px solid rgba(30,40,64,.35)}
.tt2 td:not(:first-child){text-align:right}
.tt2 tr:last-child td{border-bottom:none}
.pr{display:flex;align-items:center;gap:.35rem;justify-content:flex-end}
.pb2{height:3px;border-radius:2px;opacity:.5}
.prgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(125px,1fr));gap:.45rem}
.pri{background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:6px;padding:.38rem .65rem}
.prk{font-size:.57rem;color:var(--muted);margin-bottom:.18rem}
.prv{font-size:.76rem;color:var(--accent2);font-weight:500}
.dc{display:flex;align-items:flex-end;gap:3px;height:50px;margin-top:.45rem}
.dw{flex:1;display:flex;flex-direction:column;align-items:center}
.db{width:100%;border-radius:2px 2px 0 0;min-height:2px}
.dl{font-size:.48rem;color:var(--muted);margin-top:.18rem;white-space:nowrap}
</style>
</head>
<body>
<div class="app">
<header>
  <div class="logo"><div class="dot"></div>Swing Edge</div>
  <div class="hm" id="gt"></div>
</header>
<main>
  <div class="ab" id="ab"><div class="ab-h">🚨 Señales de compra activas</div><div class="ac" id="ac"></div></div>
  <div><div class="st">Resumen global</div><div class="sr" id="sr"></div></div>
  <div>
    <div class="st">Universo de activos optimizados</div>
    <div class="oos-note">⚡ Métricas OOS — período de validación fuera de muestra (parámetros nunca vistos)</div>
    <div class="tw"><table>
      <thead><tr><th>Activo</th><th>Win% OOS</th><th>PF</th><th>Sharpe</th><th>Avg W</th><th>Avg L</th><th>≥3%</th><th>Max DD</th><th>Total OOS</th><th>Trades OOS</th><th>Estado</th></tr></thead>
      <tbody id="atb"></tbody>
    </table></div>
  </div>
  <div class="dp" id="dp">
    <div class="ph">
      <div><div class="ptk" id="ptk">—</div><div id="ptn" style="color:var(--muted);font-size:.7rem;margin-top:.18rem"></div></div>
      <button class="pc" onclick="closePanel()">✕</button>
    </div>
    <div class="pb">
      <div style="margin-bottom:1rem"><span class="oos-note">⚡ Métricas OOS · Historial completo de trades abajo</span></div>
      <div class="ms" id="mss"></div>
      <div class="pg">
        <div><div class="st" style="margin-bottom:.7rem">Precio 90d + EMAs</div><div class="cw"><canvas id="pc2"></canvas></div></div>
        <div><div class="st" style="margin-bottom:.7rem">RSI · ADX</div><div class="cw"><canvas id="rc2"></canvas></div></div>
      </div>
      <div style="margin-bottom:1.4rem"><div class="st" style="margin-bottom:.55rem">Distribución P&L (historial completo)</div><div class="dc" id="dcc"></div></div>
      <div style="margin-bottom:1.4rem">
        <div class="st" style="margin-bottom:.7rem">Historial completo de trades</div>
        <div style="overflow-x:auto"><table class="tt2">
          <thead><tr><th>Entrada</th><th>Salida</th><th>Ent$</th><th>Sal$</th><th>SL</th><th>TP</th><th>P&L</th><th>Días</th><th>Razón</th></tr></thead>
          <tbody id="ttb"></tbody>
        </table></div>
      </div>
      <div><div class="st" style="margin-bottom:.7rem">Parámetros óptimos</div><div class="prgrid" id="prg"></div></div>
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
  const pad={t:8,r:8,b:22,l:42},cW=W-pad.l-pad.r,cH=H-pad.t-pad.b;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='rgba(30,40,64,.5)';ctx.lineWidth=1;
  for(let g=0;g<=4;g++){const y=pad.t+g/4*cH;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}
  let all=datasets.flatMap(d=>d.data.filter(v=>v!=null&&!isNaN(v)));
  let mn=opts.min??Math.min(...all),mx=opts.max??Math.max(...all);
  if(mx===mn){mx+=1;mn-=1;}const rng=mx-mn;
  const xp=(i,n)=>pad.l+i/(n-1)*cW,yp=v=>pad.t+(1-(v-mn)/rng)*cH;
  (opts.refs||[]).forEach(r=>{ctx.strokeStyle=r.c||'rgba(255,255,255,.15)';ctx.lineWidth=1;ctx.setLineDash([4,4]);const y=yp(r.v);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=r.c||'rgba(255,255,255,.4)';ctx.font='8px JetBrains Mono';ctx.fillText(r.l||r.v,W-pad.r-24,y-2);});
  datasets.forEach(ds=>{
    const data=ds.data,n=data.length;if(!n)return;
    ctx.strokeStyle=ds.c||'#00e5a0';ctx.lineWidth=ds.w||1.5;ctx.setLineDash(ds.d||[]);
    if(ds.fill){ctx.beginPath();let s=false;data.forEach((v,i)=>{if(v==null)return;const x=xp(i,n),y=yp(v);s?ctx.lineTo(x,y):(ctx.moveTo(x,y),s=true);});ctx.lineTo(xp(n-1,n),pad.t+cH);ctx.lineTo(pad.l,pad.t+cH);ctx.closePath();ctx.fillStyle=ds.fill;ctx.fill();}
    ctx.beginPath();let s=false;data.forEach((v,i)=>{if(v==null||isNaN(v))return;const x=xp(i,n),y=yp(v);s?ctx.lineTo(x,y):(ctx.moveTo(x,y),s=true);});ctx.stroke();ctx.setLineDash([]);
    if(ds.sigs)ds.sigs.forEach((sg,i)=>{if(sg&&data[i]!=null){const x=xp(i,n),y=yp(data[i]);ctx.fillStyle='#00e5a0';ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();}});
  });
  ctx.fillStyle='rgba(90,106,138,.8)';ctx.font='8px JetBrains Mono';
  for(let g=0;g<=4;g++){const v=mn+(1-g/4)*rng,y=pad.t+g/4*cH;ctx.fillText(v.toFixed(opts.dec??0),2,y+3);}
  if(opts.dates){const step=Math.ceil(opts.dates.length/5);opts.dates.forEach((d,i)=>{if(i%step!==0)return;ctx.fillStyle='rgba(90,106,138,.7)';ctx.fillText(d.slice(5),xp(i,opts.dates.length)-10,H-5);});}
}

if(D.generated_at){const d=new Date(D.generated_at);document.getElementById('gt').textContent='Actualizado: '+d.toLocaleString('es-ES',{dateStyle:'short',timeStyle:'short'});}

const al=D.alerts||[];
if(al.length){
  document.getElementById('ab').classList.add('on');
  document.getElementById('ac').innerHTML=al.map(a=>`
    <div class="acard ${a.days_ago===0?'today':''}" onclick="openAsset('${a.ticker}')">
      <div class="ar"><div class="at">${a.ticker}</div><span class="au ${a.days_ago===0?'today':''}">${a.urgency}</span></div>
      <div class="an">${a.name||''}</div>
      <div style="font-size:.84rem;font-weight:500">$${a.price?.toFixed(2)}</div>
      <div class="stgrid">
        <div class="slb"><div class="bl">STOP LOSS</div><div>$${a.stop_loss?.toFixed(2)} <span style="opacity:.6">-${a.stop_pct}%</span></div></div>
        <div class="tpb"><div class="bl">TAKE PROFIT</div><div>$${a.take_profit?.toFixed(2)} <span style="opacity:.6">+${a.tp_pct}%</span></div></div>
      </div>
      <div class="rrb" style="color:${a.rr_ratio>=1.5?'var(--green)':a.rr_ratio>=1.0?'var(--yellow)':'var(--red)'}">
        ⚖ R/R ${a.rr_ratio??'?'}:1
        <span style="opacity:.6;font-size:.6rem">${a.rr_ratio>=1.5?' bueno ✓':a.rr_ratio>=1.0?' ajustado ⚠':' bajo ✗'}</span>
      </div>
      <div style="font-size:.63rem;color:var(--muted);margin-top:.4rem;line-height:1.8">
        <span style="color:var(--accent2)">Trailing:</span> activa tras +${a.trail_act??'?'}% → stop ${a.trail_atr??'?'}×ATR del máximo<br>
        <span style="color:var(--accent2)">Tiempo máx:</span> ${a.max_days??'?'} días hábiles
      </div>
      <div class="chips">
        ${a.rsi?`<span class="chip">RSI ${a.rsi}</span>`:''}
        ${a.adx?`<span class="chip">ADX ${a.adx}</span>`:''}
        ${a.vol_ratio?`<span class="chip">Vol ×${a.vol_ratio}</span>`:''}
        <span class="chip">Score ${a.score}/100</span>
      </div>
    </div>`).join('');
  const note=document.createElement('div');
  note.style.cssText='font-size:.62rem;color:var(--muted);margin-top:.8rem;padding-top:.6rem;border-top:1px solid rgba(255,255,255,.06)';
  note.innerHTML='⚖ <b style="color:var(--text)">R/R</b>: cuánto ganas por cada euro arriesgado. <span style="color:var(--green)">≥1.5 bueno</span> · <span style="color:var(--yellow)">1.0-1.5 ajustado</span> · <span style="color:var(--red)">&lt;1.0 desfavorable</span>';
  document.getElementById('ab').appendChild(note);
}

// Global stats — usa métricas OOS
const assets=Object.values(D.assets||{});
let nt=0,nw=0,tp=0,na=0,ss=[],ps=[];
assets.forEach(a=>{
  const m=a.metrics_oos||{};
  if(m.n){nt+=m.n;nw+=Math.round((m.wr||0)/100*m.n);tp+=m.total||0;if(m.sharpe)ss.push(m.sharpe);if(m.pf)ps.push(m.pf);}
  if(a.alert)na++;
});
const ash=ss.length?ss.reduce((a,b)=>a+b)/ss.length:0,apf=ps.length?ps.reduce((a,b)=>a+b)/ps.length:0,wr=nt?nw/nt*100:0;
const stats=[
  {l:'Activos',v:assets.length,c:'b',s:'en universo'},
  {l:'Alertas activas',v:na,c:na?'r':'m',s:'señales de compra'},
  {l:'Win Rate OOS',v:wr.toFixed(1)+'%',c:wr>=50?'g':'y',s:`${nw}/${nt} trades OOS`},
  {l:'Profit Factor',v:apf.toFixed(2),c:apf>=1.5?'g':'y',s:'promedio OOS'},
  {l:'Sharpe',v:ash.toFixed(2),c:ash>=1?'g':'y',s:'promedio OOS'},
  {l:'Retorno OOS',v:(tp>=0?'+':'')+tp.toFixed(1)+'%',c:tp>=0?'g':'r',s:'suma OOS'},
];
document.getElementById('sr').innerHTML=stats.map(s=>`<div class="sc"><div class="sl">${s.l}</div><div class="sv ${s.c}">${s.v}</div><div class="ss">${s.s}</div></div>`).join('');

// Asset table — usa métricas OOS
assets.sort((a,b)=>((b.metrics_oos||{}).sharpe||0)-((a.metrics_oos||{}).sharpe||0));
document.getElementById('atb').innerHTML=assets.map(a=>{
  const m=a.metrics_oos||{},ha=!!a.alert;
  const wc=m.wr>=55?'bg':m.wr>=45?'by':'br';
  const pc=m.pf>=1.5?'bg':m.pf>=1?'by':'br';
  const sc=m.sharpe>=1?'bg':m.sharpe>=0.5?'by':'br';
  const tc=m.total>=0?'pos':'neg';
  return `<tr onclick="openAsset('${a.ticker}')">
    <td><div class="tt">${a.ticker}</div><div class="tn">${a.name||''}</div></td>
    <td><span class="bdg ${wc}">${m.wr?.toFixed(1)||'—'}%</span></td>
    <td><span class="bdg ${pc}">${m.pf?.toFixed(2)||'—'}</span></td>
    <td><span class="bdg ${sc}">${m.sharpe?.toFixed(2)||'—'}</span></td>
    <td class="pos">+${m.avg_w?.toFixed(2)||'—'}%</td>
    <td class="neg">${m.avg_l?.toFixed(2)||'—'}%</td>
    <td>${m.p3?.toFixed(0)||'—'}%</td>
    <td class="neg">${m.dd?.toFixed(1)||'—'}%</td>
    <td class="${tc}">${m.total>=0?'+':''}${m.total?.toFixed(1)||'—'}%</td>
    <td>${m.n||'—'}</td>
    <td>${ha?'<span class="bdg ba">🚨 SEÑAL</span>':'<span class="bdg bg">OK</span>'}</td>
  </tr>`;
}).join('');

function openAsset(ticker){
  const a=D.assets[ticker];if(!a)return;
  document.getElementById('ptk').textContent=ticker;
  document.getElementById('ptn').textContent=a.name||'';
  // Panel usa métricas OOS
  const m=a.metrics_oos||{},p=a.params||{};
  const minis=[
    {v:(m.wr||0).toFixed(1)+'%',l:'Win Rate OOS',c:m.wr>=50?'var(--green)':'var(--yellow)'},
    {v:(m.pf||0).toFixed(2),l:'Profit Factor',c:m.pf>=1.5?'var(--green)':'var(--yellow)'},
    {v:(m.sharpe||0).toFixed(2),l:'Sharpe',c:m.sharpe>=1?'var(--green)':'var(--yellow)'},
    {v:'+'+(m.avg_w||0).toFixed(2)+'%',l:'Avg Win OOS',c:'var(--green)'},
    {v:(m.avg_l||0).toFixed(2)+'%',l:'Avg Loss OOS',c:'var(--red)'},
    {v:(m.dd||0).toFixed(1)+'%',l:'Max DD OOS',c:'var(--red)'},
    {v:(m.p3||0).toFixed(0)+'%',l:'≥3% trades',c:m.p3>=30?'var(--green)':'var(--yellow)'},
    {v:(m.p5||0).toFixed(0)+'%',l:'≥5% trades',c:'var(--accent2)'},
    {v:(m.n||0).toString(),l:'Trades OOS',c:'var(--accent2)'},
  ];
  document.getElementById('mss').innerHTML=minis.map(s=>`<div class="ms-c"><div class="mv" style="color:${s.c}">${s.v}</div><div class="ml">${s.l}</div></div>`).join('');
  // Historial completo de trades
  const trades=a.trades||[];
  document.getElementById('ttb').innerHTML=trades.map(t=>{
    const c=t.pnl>=0?'var(--green)':'var(--red)';
    const bw=Math.min(Math.abs(t.pnl)*4,60);
    const stars=t.pnl>=5?' ★★':t.pnl>=3?' ★':'';
    return `<tr>
      <td>${t.entry_date}</td><td>${t.exit_date}</td>
      <td>$${t.entry_price}</td><td>$${t.exit_price}</td>
      <td style="color:var(--red)">$${t.stop_loss}</td>
      <td style="color:var(--green)">$${t.take_profit}</td>
      <td><div class="pr"><span style="color:${c}">${t.pnl>=0?'+':''}${t.pnl.toFixed(2)}%${stars}</span><div class="pb2" style="width:${bw}px;background:${c}"></div></div></td>
      <td>${t.days}d</td>
      <td style="font-size:.6rem;color:var(--muted)">${t.reason}</td>
    </tr>`;
  }).join('');
  const kp=['rsi_p','rsi_lo','rsi_hi','ema_f','ema_s','ema_t','macd_f','macd_s','adx_min','score_min','tp_pct','atr_stop','max_days','vol_min','trail_act','trail_atr'];
  document.getElementById('prg').innerHTML=kp.filter(k=>p[k]!==undefined).map(k=>`<div class="pri"><div class="prk">${k}</div><div class="prv">${p[k]}</div></div>`).join('');
  if(trades.length){
    const pnls=trades.map(t=>t.pnl);
    const bins=[[-99,-10],[-10,-5],[-5,-3],[-3,0],[0,3],[3,5],[5,10],[10,99]];
    const lbls=['<-10','-10→-5','-5→-3','-3→0','0→3','3→5','5→10','>10'];
    const cnts=bins.map(([lo,hi])=>pnls.filter(p=>p>lo&&p<=hi).length);
    const mx=Math.max(...cnts,1);
    document.getElementById('dcc').innerHTML=cnts.map((c,i)=>{
      const h=Math.round(c/mx*46),lo=bins[i][0];
      const col=lo>=3?'var(--green)':lo>=0?'rgba(0,229,160,.35)':lo>=-5?'rgba(255,71,87,.45)':'var(--red)';
      return `<div class="dw"><div class="db" style="height:${h}px;background:${col}"></div><div class="dl">${lbls[i]}</div></div>`;
    }).join('');
  }
  document.getElementById('dp').classList.add('open');
  setTimeout(()=>{
    document.getElementById('dp').scrollIntoView({behavior:'smooth',block:'start'});
    const h=a.price_history||[];
    if(h.length){
      drawChart('pc2',[
        {data:h.map(x=>x.close),c:'#8899bb',w:1.5,fill:'rgba(77,159,255,.04)',sigs:h.map(x=>x.signal===1)},
        {data:h.map(x=>x.ema_f),c:'#ffb300',w:1.2},
        {data:h.map(x=>x.ema_s),c:'#00e5a0',w:1.5},
        {data:h.map(x=>x.ema_t),c:'#9d6aff',w:1,d:[4,4]},
      ],{dec:1,dates:h.map(x=>x.date)});
      drawChart('rc2',[
        {data:h.map(x=>x.rsi),c:'#ff4757',w:1.5},
        {data:h.map(x=>x.adx),c:'#4d9fff',w:1.2,d:[3,3]},
      ],{min:0,max:100,dec:0,dates:h.map(x=>x.date),refs:[
        {v:70,c:'rgba(255,71,87,.4)',l:'70'},{v:50,c:'rgba(255,255,255,.12)',l:'50'},
        {v:30,c:'rgba(0,229,160,.3)',l:'30'},{v:25,c:'rgba(77,159,255,.3)',l:'ADX25'},
      ]});
    }
  },60);
}
function closePanel(){document.getElementById('dp').classList.remove('open');}
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

    print("  Descargando SPY...")
    spy = yf.download("SPY", period=PERIOD, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns=spy.columns.get_level_values(0)
    spy_c = spy['Close'].squeeze()
    spy_e = spy_c.ewm(span=50, adjust=False).mean()

    tickers = [TICKER_SOLO] if TICKER_SOLO else list(cache.keys())
    alerts   = []
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

        macro = (spy_c > spy_e).reindex(df_raw.index, method='ffill').fillna(False).values

        use_invert = entry.get('invert', ticker in INVERSE_TICKERS or ticker in VIX_TICKERS)
        ind    = calc_ind(df_raw, p, use_invert)
        sigs   = get_signals(ind, p, macro, use_invert)
        scores = np.array([score_bar(ind, i, p) if i>=35 else 0
                           for i in range(len(ind['c']))])

        # Historial completo de trades (para ver entradas/salidas)
        trades = build_rich_trades(ind, sigs, p, ticker)

        # ── Terminal: mostrar métricas OOS ─────────────────────
        if not SOLO_ALERTAS and m_oos:
            print_summary(ticker, name, m_oos, p)
            print_trades(trades, ticker)

        alert = check_alert(ind, sigs, p, ticker, name, trades)
        if alert:
            alerts.append(alert)
            print_alert(alert)

        price_hist = build_price_history(ind, sigs, scores)
        all_data[ticker] = {
            "ticker":        ticker,
            "name":          name,
            "params":        p,
            "metrics_oos":   m_oos,   # métricas honestas para dashboard
            "trades":        trades,  # historial completo para referencia
            "price_history": price_hist,
            "alert":         alert,
            "optimized_at":  entry.get('optimized_at','')[:10],
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
        "generated_at": datetime.now().isoformat(),
        "alerts":        alerts,
        "assets":        all_data,
    }
    OUTPUT_FILE.write_text(json.dumps(dashboard_data, ensure_ascii=False, indent=2))
    _generate_dashboard(dashboard_data, BASE_DIR / "dashboard.html")

    print(f"\n  {G}→ dashboard_data.json generado{RST}")
    print(f"  {G}→ dashboard.html generado (abre directamente en el navegador){RST}\n")

if __name__ == "__main__":
    main()

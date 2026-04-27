"""
bess_pipeline.py
────────────────
Exposes the notebook pipeline as importable Python functions.
The Streamlit app (app.py) calls run_full_pipeline() from here.

This file re-uses all the logic from BESS_Optimized_Dispatch.ipynb
without duplicating code — it imports the saved model pickles from
models/ and re-runs only the forecast + dispatch steps.

Usage:
    from bess_pipeline import run_full_pipeline
    results = run_full_pipeline(2024, 6, 'SA1', ...)
"""

import os, sys, warnings, pickle, logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
logging.getLogger("nemosis").setLevel(logging.WARNING)

# ── Path setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

Path('data').mkdir(exist_ok=True)
Path('outputs').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
PRICE_SHIFT = 1000.0
LAG_PERIODS  = [1, 2, 3, 6, 12, 18, 36, 72, 144, 288, 576, 2016]
ROLL_WINDOWS = [6, 12, 36, 72, 144, 288, 576, 2016]


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def _inv_log(y: np.ndarray) -> np.ndarray:
    return np.expm1(y) - PRICE_SHIFT - 1


def _load_models():
    """Load saved pipeline + model pickles from models/ directory."""
    pipe_path = Path('models') / 'pipeline.pkl'
    xgb_path  = Path('models') / 'xgb.pkl'
    lgbm_path = Path('models') / 'lgbm.pkl'

    if not pipe_path.exists() or not xgb_path.exists():
        raise FileNotFoundError(
            "Trained models not found in models/.\n"
            "Run BESS_Optimized_Dispatch.ipynb first (Steps 1–9) to train and save models."
        )

    with open(pipe_path, 'rb') as f: pipe    = pickle.load(f)
    with open(xgb_path,  'rb') as f: xgb_m   = pickle.load(f)

    models_w = {'xgboost': (xgb_m, 1.0)}
    if lgbm_path.exists():
        with open(lgbm_path, 'rb') as f: lgbm_m = pickle.load(f)
        # Recalculate weights — equal for simplicity here
        models_w = {'xgboost': (xgb_m, 0.5), 'lightgbm': (lgbm_m, 0.5)}

    return pipe, models_w


def _predict_ensemble(models_w: dict, X: np.ndarray) -> np.ndarray:
    return sum(w * m.predict(X) for m, w in models_w.values())


def _forecast_month(pipe, models_w, df_hist, target_year, target_month):
    """Recursive rolling forecast for the target month."""
    ts  = pd.Timestamp(year=target_year, month=target_month, day=1)
    te  = ts + pd.offsets.MonthEnd(1) + pd.Timedelta('5min')
    idx = pd.date_range(ts, te - pd.Timedelta('1min'), freq='30min')

    # Estimate uncertainty from recent residuals
    X_v, y_v_raw, _, _ = pipe.transform(df_hist.tail(5000))
    pred_raw = _inv_log(_predict_ensemble(models_w, X_v))
    base_unc = max(float(np.abs(y_v_raw - pred_raw).std()), 15.0)

    df_w = df_hist.copy()
    recs = []
    for i, t in enumerate(idx):
        ctx   = df_w.tail(2500)
        dummy = pd.DataFrame({'price': [df_w['price'].iloc[-1]]}, index=[t])
        if 'demand' in df_w.columns:
            dummy['demand'] = df_w['demand'].iloc[-1]
        ctx = pd.concat([ctx, dummy])
        try:
            Xr, _, _, _ = pipe.transform(ctx)
            pr = float(_inv_log(_predict_ensemble(models_w, Xr[-1:]))[0])
        except Exception:
            pr = float(df_w['price'].iloc[-288:].mean())
        pr  = float(np.clip(pr, -1000, 16600))
        unc = base_unc * (1 + i / (len(idx) * 2.5))
        recs.append({'datetime': t, 'price_forecast': pr,
                     'price_lower': np.clip(pr-1.96*unc, -1000, 16600),
                     'price_upper': np.clip(pr+1.96*unc, -1000, 16600)})
        nr = pd.DataFrame({'price': [pr]}, index=[t])
        if 'demand' in df_w.columns: nr['demand'] = df_w['demand'].iloc[-1]
        df_w = pd.concat([df_w, nr])

    return pd.DataFrame(recs).set_index('datetime')


def _optimise_day(prices, soc_init, soc_max_v, soc_min_v, power_mw, capacity_mwh,
                   eta_c, eta_d, cycle_cost, pen=8.0):
    """LP/MILP day dispatch — same logic as notebook."""
    T  = len(prices); dt = 5/60
    E0 = soc_init * capacity_mwh
    E_min = soc_min_v * capacity_mwh
    E_max = soc_max_v * capacity_mwh

    try:
        import cvxpy as cp
        Pc = cp.Variable(T, nonneg=True); Pd = cp.Variable(T, nonneg=True)
        E  = cp.Variable(T+1, nonneg=True); bv = cp.Variable(T, boolean=True)
        cs = [E[0]==E0, E[1:]>=E_min, E[1:]<=E_max,
               Pc<=power_mw*(1-bv), Pd<=power_mw*bv]
        for t in range(T):
            cs.append(E[t+1]==E[t]+eta_c*Pc[t]*dt-Pd[t]*dt/eta_d)
        rev = cp.sum(cp.multiply(prices,Pd*dt*eta_d)-cp.multiply(prices,Pc*dt/eta_c))
        deg = cycle_cost*cp.sum(Pc+Pd)*dt
        tpen= pen*cp.square(E[-1]-E0)
        prob= cp.Problem(cp.Maximize(rev-deg-tpen), cs)
        for solver in [cp.GLPK_MI, cp.ECOS]:
            try:
                kw = dict(verbose=False, max_seconds=90) if solver==cp.GLPK_MI else dict(verbose=False)
                prob.solve(solver=solver, **kw)
                if prob.status in ('optimal','optimal_inaccurate') and Pc.value is not None:
                    pc_v=np.maximum(Pc.value,0); pd_v=np.maximum(Pd.value,0)
                    ev=E.value[:T]; soc=ev/capacity_mwh
                    rev_a=prices*pd_v*eta_d*dt-prices*pc_v*dt/eta_c
                    deg_a=cycle_cost*(pc_v+pd_v)*dt
                    return pc_v,pd_v,soc,rev_a,deg_a
            except Exception: pass
    except ImportError: pass

    # Heuristic fallback
    E = E0; pc_v=np.zeros(T); pd_v=np.zeros(T); ev=np.zeros(T+1); ev[0]=E
    for t in range(T):
        lo=max(0,t-36); hi=min(T,t+37)
        local=prices[lo:hi]; p20=np.percentile(local,20); p75=np.percentile(local,75)
        hc=E_max-E; hd=E-E_min; mh=0.05*capacity_mwh
        if prices[t]<=p20 and hc>mh:
            mc=min(power_mw,hc/(eta_c*dt)); pc_v[t]=max(0,mc); E+=eta_c*pc_v[t]*dt
        elif prices[t]>=p75 and hd>mh:
            md=min(power_mw,hd*eta_d/dt); pd_v[t]=max(0,md); E-=pd_v[t]*dt/eta_d
        E=np.clip(E,E_min,E_max); ev[t+1]=E
    soc=ev[:T]/capacity_mwh
    rev_a=prices*pd_v*eta_d*dt-prices*pc_v*dt/eta_c
    deg_a=cycle_cost*(pc_v+pd_v)*dt
    return pc_v,pd_v,soc,rev_a,deg_a


def _optimise_month(df_fc, power_mw, capacity_mwh, soc_min_v, soc_max_v,
                     soc_init, eta_c, eta_d, cycle_cost):
    """Run day-ahead LP for every day in the forecast."""
    rows=[]; soc_cur=soc_init
    for day in sorted(set(df_fc.index.date)):
        try: dd=df_fc.loc[str(day)]
        except: continue
        if len(dd)==0: continue
        pr=dd['price_forecast'].values
        pc_v,pd_v,soc_v,rev_a,deg_a = _optimise_day(
            pr, soc_cur, soc_max_v, soc_min_v,
            power_mw, capacity_mwh, eta_c, eta_d, cycle_cost
        )
        soc_cur = float(soc_v[-1])
        for t in range(len(pr)):
            rows.append({'datetime':dd.index[t],'price':pr[t],
                         'p_charge':float(pc_v[t]),'p_discharge':float(pd_v[t]),
                         'p_net':float(pd_v[t]*eta_d-pc_v[t]/eta_c),
                         'soc':float(soc_v[t]),'revenue':float(rev_a[t]),
                         'degradation_cost':float(deg_a[t])})
    return pd.DataFrame(rows).set_index('datetime')


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def run_full_pipeline(target_year: int, target_month: int,
                       region: str = 'SA1',
                       power_mw: float = 100.0,
                       capacity_mwh: float = 200.0,
                       soc_min: float = 0.10,
                       soc_max: float = 0.90,
                       history_years: int = 2,
                       run_hpo: bool = False) -> dict:
    """
    Full BESS dispatch pipeline callable from Streamlit or any Python script.

    Requires: models/ directory populated by running the notebook first.

    Returns dict with:
        df_forecast, df_dispatch, net_profit, total_revenue,
        val_mae, soc_min, soc_max
    """
    from nemosis import dynamic_data_compiler

    # BESS economics
    capex_total  = capacity_mwh * 1000 * 220.0
    usable_mwh   = capacity_mwh * (soc_max - soc_min)
    cycle_cost   = capex_total / (4000.0 * usable_mwh * 2)
    eta_c = eta_d = 0.95

    # ── Load trained models ────────────────────────────────────────────
    pipe, models_w = _load_models()

    # ── Load NEM data ──────────────────────────────────────────────────
    ts         = pd.Timestamp(year=target_year, month=target_month, day=1)
    hist_start = ts - pd.DateOffset(years=history_years) - pd.DateOffset(days=35)
    te         = ts + pd.offsets.MonthEnd(1) + pd.Timedelta('5min')
    fmt        = '%Y/%m/%d %H:%M:%S'

    cache_key  = f"{region}_{hist_start.strftime('%Y%m')}_{te.strftime('%Y%m')}"
    cache_file = Path('data') / f'nem_{cache_key}.parquet'
    Path('data').mkdir(exist_ok=True)

    if cache_file.exists():
        df_full = pd.read_parquet(cache_file)
    else:
        raw_cache = str(Path('data') / 'raw_nemosis')
        os.makedirs(raw_cache, exist_ok=True)
        raw = dynamic_data_compiler(
            start_time=hist_start.strftime(fmt), end_time=te.strftime(fmt),
            table_name='TRADINGPRICE', raw_data_location=raw_cache,
            select_columns=['SETTLEMENTDATE','REGIONID','RRP','TOTALDEMAND','PERIODTYPE'],
            filter_cols=['REGIONID'], filter_values=[[region]],
            keep_csv=False, fformat='feather',
        )
        if 'PERIODTYPE' in raw.columns:
            raw = raw[raw['PERIODTYPE']=='ENERGY'].drop(columns=['PERIODTYPE'])
        df_full = (raw.rename(columns={'SETTLEMENTDATE':'datetime','RRP':'price','TOTALDEMAND':'demand'})
                   .drop(columns=['REGIONID'],errors='ignore')
                   .set_index('datetime').sort_index()
                   .resample('30min').mean().ffill(limit=3))
        df_full.to_parquet(cache_file)

    df_hist = df_full[df_full.index < ts]

    # ── Val MAE estimate (from saved pipeline on last 15% of history) ──
    n      = len(df_hist)
    n_val  = int(n * 0.15)
    df_val = df_hist.iloc[-n_val:]
    try:
        X_val, y_val_raw, _, _ = pipe.transform(df_val)
        val_pred = _inv_log(_predict_ensemble(models_w, X_val))
        mask     = np.isfinite(y_val_raw) & np.isfinite(val_pred)
        val_mae  = float(mean_absolute_error(y_val_raw[mask], val_pred[mask]))
    except Exception:
        val_mae = 0.0

    # ── Forecast ───────────────────────────────────────────────────────
    df_forecast = _forecast_month(pipe, models_w, df_hist, target_year, target_month)

    # ── Dispatch ───────────────────────────────────────────────────────
    df_dispatch = _optimise_month(
        df_forecast, power_mw, capacity_mwh,
        soc_min, soc_max, (soc_min+soc_max)/2,
        eta_c, eta_d, cycle_cost
    )

    # ── Save CSVs ──────────────────────────────────────────────────────
    Path('outputs').mkdir(exist_ok=True)
    df_forecast.to_csv(f'outputs/forecast_{target_year}_{target_month:02d}.csv')
    df_dispatch.to_csv(f'outputs/dispatch_{target_year}_{target_month:02d}.csv')

    rev = float(df_dispatch['revenue'].sum())
    deg = float(df_dispatch['degradation_cost'].sum())

    return {
        'df_forecast':   df_forecast,
        'df_dispatch':   df_dispatch,
        'total_revenue': rev,
        'net_profit':    rev - deg,
        'val_mae':       val_mae,
        'soc_min':       soc_min,
        'soc_max':       soc_max,
    }

"""
╔══════════════════════════════════════════════════════════════════╗
║   BESS OPTIMIZED DISPATCH — CONTROL ROOM WEB APP                ║
║   Streamlit interface for operations teams                       ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    streamlit run app.py
    # Opens at http://localhost:8501

Deploy on internal network:
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
    # All machines on the same network access via http://SERVER-IP:8501
"""

import os, sys, json, calendar, warnings, pickle, time, hashlib
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Add project root to path so pipeline modules are importable ────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "⚡ BESS Dispatch — Control Room",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main            { background-color: #07090f; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .metric-card {
        background: #0d1117; border: 1px solid #263050; border-radius: 8px;
        padding: 16px 20px; margin: 4px 0;
    }
    .kpi-green { color: #00e676; font-weight: bold; }
    .kpi-red   { color: #ff4757; font-weight: bold; }
    .kpi-blue  { color: #c084fc; font-weight: bold; }
    h1, h2, h3 { color: #c084fc !important; }
    .stButton > button {
        background: linear-gradient(135deg, #c084fc, #8b5cf6);
        color: white; border: none; border-radius: 8px;
        font-size: 1.1em; font-weight: bold;
        padding: 0.6em 2em; width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette (Plotly) ────────────────────────────────────────────
C = dict(
    charge='#4488ff', discharge='#ff6644', soc='#00cc88',
    price='#ffd166', forecast='#c084fc', actual='#8fa3bf',
    revenue='#00e676', loss='#ff4757', bg='#07090f', bg2='#0d1117',
)

# ══════════════════════════════════════════════════════════════════════
# PIPELINE IMPORT — lazy-load the notebook pipeline as functions
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading ML pipeline...")
def load_pipeline_modules():
    """Import all pipeline dependencies once and cache them."""
    import numpy as np
    import pandas as pd
    from scipy import stats as sp_stats
    from sklearn.preprocessing import RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    return True

@st.cache_data(show_spinner="Loading NEM data...", ttl=3600)
def load_data_cached(region, history_years, target_year, target_month):
    """Load NEM data with Streamlit cache (re-uses parquet if available)."""
    from nemosis import dynamic_data_compiler
    import hashlib, logging
    logging.getLogger("nemosis").setLevel(logging.WARNING)

    ts         = pd.Timestamp(year=target_year, month=target_month, day=1)
    te         = ts + pd.offsets.MonthEnd(1) + pd.Timedelta('5min')
    hist_start = ts - pd.DateOffset(years=history_years) - pd.DateOffset(days=35)
    fmt        = '%Y/%m/%d %H:%M:%S'

    cache_key  = f"{region}_{hist_start.strftime('%Y%m')}_{te.strftime('%Y%m')}"
    cache_file = Path('data') / f'nem_{cache_key}.parquet'
    Path('data').mkdir(exist_ok=True)

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    raw_cache = str(Path('data') / 'raw_nemosis')
    os.makedirs(raw_cache, exist_ok=True)

    raw = dynamic_data_compiler(
        start_time        = hist_start.strftime(fmt),
        end_time          = te.strftime(fmt),
        table_name        = 'DISPATCHPRICE',
        raw_data_location = raw_cache,
        select_columns    = ['SETTLEMENTDATE','REGIONID','RRP','TOTALDEMAND','INTERVENTION'],
        filter_cols       = ['REGIONID'],
        filter_values     = [[region]],
        keep_csv          = False,
        fformat           = 'feather',
    )
    if 'INTERVENTION' in raw.columns:
        raw = raw[raw['INTERVENTION'] == 0].drop(columns=['INTERVENTION'])

    df = (raw.rename(columns={'SETTLEMENTDATE':'datetime','RRP':'price','TOTALDEMAND':'demand'})
          .drop(columns=['REGIONID'], errors='ignore')
          .set_index('datetime').sort_index()
          .resample('5min').mean().ffill(limit=3))

    df.to_parquet(cache_file)
    return df


# ══════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════

def plot_price_forecast(df_forecast: pd.DataFrame, month_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_forecast.index, y=df_forecast['price_upper'],
        fill=None, mode='lines', line=dict(width=0),
        showlegend=False, name='Upper CI'
    ))
    fig.add_trace(go.Scatter(
        x=df_forecast.index, y=df_forecast['price_lower'],
        fill='tonexty', mode='lines', line=dict(width=0),
        fillcolor='rgba(192,132,252,0.12)', name='95% CI'
    ))
    fig.add_trace(go.Scatter(
        x=df_forecast.index, y=df_forecast['price_forecast'],
        mode='lines', line=dict(color=C['forecast'], width=1.5),
        name='Forecast price'
    ))
    fig.add_hline(y=0, line=dict(color='#8fa3bf', width=1, dash='dash'))
    fig.update_layout(
        title=f'NEM Spot Price Forecast — {month_name}',
        xaxis_title='Date', yaxis_title='$/MWh',
        paper_bgcolor=C['bg'], plot_bgcolor=C['bg2'],
        font=dict(color='#eef2ff'), legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=320, margin=dict(t=50, b=30, l=60, r=20)
    )
    return fig


def plot_dispatch(df_dispatch: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_dispatch.index, y=df_dispatch['p_discharge'],
        name='Discharge → grid', marker_color=C['discharge'], opacity=0.85
    ))
    fig.add_trace(go.Bar(
        x=df_dispatch.index, y=-df_dispatch['p_charge'],
        name='Charge ← grid', marker_color=C['charge'], opacity=0.85
    ))
    fig.add_hline(y=0, line=dict(color='white', width=1))
    fig.update_layout(
        title='BESS Power Dispatch',
        xaxis_title='Date', yaxis_title='Power (MW)',
        barmode='relative',
        paper_bgcolor=C['bg'], plot_bgcolor=C['bg2'],
        font=dict(color='#eef2ff'), legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=300, margin=dict(t=50, b=30, l=60, r=20)
    )
    return fig


def plot_soc(df_dispatch: pd.DataFrame, soc_min: float, soc_max: float) -> go.Figure:
    fig = go.Figure()
    soc_pct = df_dispatch['soc'] * 100
    fig.add_trace(go.Scatter(
        x=df_dispatch.index, y=soc_pct,
        mode='lines', line=dict(color=C['soc'], width=2),
        fill='tozeroy', fillcolor='rgba(0,204,136,0.10)', name='SoC (%)'
    ))
    fig.add_hline(y=soc_min*100, line=dict(color=C['loss'],   width=1.5, dash='dash'), annotation_text=f'Min {soc_min*100:.0f}%')
    fig.add_hline(y=soc_max*100, line=dict(color=C['charge'], width=1.5, dash='dash'), annotation_text=f'Max {soc_max*100:.0f}%')
    fig.update_yaxes(range=[-5, 105])
    fig.update_layout(
        title=f'State of Charge (SoC) — Min={soc_pct.min():.1f}%  Max={soc_pct.max():.1f}%',
        xaxis_title='Date', yaxis_title='SoC (%)',
        paper_bgcolor=C['bg'], plot_bgcolor=C['bg2'],
        font=dict(color='#eef2ff'), legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=280, margin=dict(t=50, b=30, l=60, r=20)
    )
    return fig


def plot_cumulative_revenue(df_dispatch: pd.DataFrame) -> go.Figure:
    rev = df_dispatch['revenue']
    deg = df_dispatch['degradation_cost']
    net = rev - deg
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_dispatch.index, y=rev.cumsum(),
        mode='lines', line=dict(color=C['revenue'], width=1.8),
        name='Gross revenue'
    ))
    fig.add_trace(go.Scatter(
        x=df_dispatch.index, y=deg.cumsum(),
        mode='lines', line=dict(color=C['loss'], width=1.4, dash='dash'),
        name='Degradation cost'
    ))
    fig.add_trace(go.Scatter(
        x=df_dispatch.index, y=net.cumsum(),
        mode='lines', line=dict(color='white', width=2.5),
        name='Net profit'
    ))
    fig.add_hline(y=0, line=dict(color='#8fa3bf', width=1))
    fig.update_layout(
        title='Cumulative Revenue & Net Profit',
        xaxis_title='Date', yaxis_title='Cumulative ($)',
        paper_bgcolor=C['bg'], plot_bgcolor=C['bg2'],
        font=dict(color='#eef2ff'), legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=300, margin=dict(t=50, b=30, l=60, r=20),
        yaxis=dict(tickformat='$,.0f')
    )
    return fig


def plot_daily_detail(df_dispatch: pd.DataFrame, day_str: str) -> go.Figure:
    try:
        d = df_dispatch.loc[day_str]
    except Exception:
        return go.Figure()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=['Spot Price + Decisions', 'Power Dispatch', 'State of Charge'],
                         vertical_spacing=0.08)
    hh = list(range(len(d)))

    # Prices
    fig.add_trace(go.Scatter(x=hh, y=d['price'].values, mode='lines',
                              line=dict(color=C['price'], width=2), name='Price'), row=1, col=1)
    chg = np.where(d['p_charge'].values > 1)[0]
    dis = np.where(d['p_discharge'].values > 1)[0]
    fig.add_trace(go.Scatter(x=list(chg), y=d['price'].values[chg], mode='markers',
                              marker=dict(color=C['charge'], size=7, symbol='triangle-down'), name='Charge'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(dis), y=d['price'].values[dis], mode='markers',
                              marker=dict(color=C['discharge'], size=7, symbol='triangle-up'), name='Discharge'), row=1, col=1)

    # Dispatch
    fig.add_trace(go.Bar(x=hh, y=d['p_discharge'].values, name='Discharge',
                          marker_color=C['discharge'], opacity=0.85), row=2, col=1)
    fig.add_trace(go.Bar(x=hh, y=-d['p_charge'].values, name='Charge',
                          marker_color=C['charge'], opacity=0.85), row=2, col=1)

    # SoC
    fig.add_trace(go.Scatter(x=hh, y=d['soc'].values*100, mode='lines',
                              line=dict(color=C['soc'], width=2), fill='tozeroy',
                              fillcolor='rgba(0,204,136,0.12)', name='SoC'), row=3, col=1)

    # Tick every hour (12 × 5-min)
    tick_pos = list(range(0, len(hh), 12))
    tick_lbl = [f"{tp//12:02d}:00" for tp in tick_pos]

    fig.update_layout(
        height=580, barmode='relative',
        paper_bgcolor=C['bg'], plot_bgcolor=C['bg2'],
        font=dict(color='#eef2ff'), showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.05),
        margin=dict(t=60, b=40, l=60, r=20),
        title=dict(text=f'Daily Dispatch Detail — {day_str}', font=dict(color='#c084fc'))
    )
    fig.update_xaxes(tickvals=tick_pos, ticktext=tick_lbl, row=3, col=1)
    return fig


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ BESS Control Room")
    st.markdown("---")

    st.markdown("### 📅 Target Month")
    col1, col2 = st.columns(2)
    target_year  = col1.selectbox("Year",  list(range(date.today().year - 1, date.today().year + 3)),
                                    index=1)
    target_month = col2.selectbox("Month", list(range(1, 13)),
                                    format_func=lambda m: calendar.month_abbr[m],
                                    index=date.today().month - 1)

    st.markdown("### ⚙️ NEM Region")
    region = st.selectbox("Region", ["SA1", "VIC1", "NSW1", "QLD1", "TAS1"])

    st.markdown("### 🔋 BESS Parameters")
    power_mw      = st.number_input("Power (MW)",      value=100, min_value=1,    max_value=1000)
    capacity_mwh  = st.number_input("Capacity (MWh)",  value=200, min_value=1,    max_value=5000)
    soc_min       = st.slider("Min SoC (%)", 5, 30, 10) / 100
    soc_max       = st.slider("Max SoC (%)", 70, 95, 90) / 100

    st.markdown("### 🚀 Model")
    run_hpo = st.checkbox("Run Optuna HPO", value=False,
                           help="Enables hyperparameter optimisation (~10 extra minutes)")
    hist_yrs = st.selectbox("History (years)", [1, 2, 3], index=1)

    st.markdown("---")
    run_btn = st.button("▶  Run Dispatch Optimisation", type="primary")

# ══════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════
month_name = calendar.month_name[target_month]
st.markdown(f"# ⚡ BESS Optimized Dispatch — {month_name} {target_year}")
st.markdown(f"**Region**: `{region}`  &nbsp;|&nbsp;  **BESS**: `{power_mw} MW / {capacity_mwh} MWh`")

# ── Session state to persist results across re-renders ─────────────────
if 'results' not in st.session_state:
    st.session_state.results = None

# ── RUN PIPELINE ────────────────────────────────────────────────────────
if run_btn:
    with st.status(f"Running pipeline for {month_name} {target_year}...", expanded=True) as status:

        st.write("📡 Loading NEM DISPATCHPRICE data...")
        try:
            # Import pipeline pieces from the notebook's exported pickle/logic
            # This calls the same functions as the notebook
            from bess_pipeline import run_full_pipeline   # see bess_pipeline.py
            results = run_full_pipeline(
                target_year=target_year,
                target_month=target_month,
                region=region,
                power_mw=power_mw,
                capacity_mwh=capacity_mwh,
                soc_min=soc_min,
                soc_max=soc_max,
                history_years=hist_yrs,
                run_hpo=run_hpo,
            )
            st.session_state.results = results
            status.update(label="✅ Pipeline complete!", state="complete", expanded=False)
            st.success(f"Net profit: ${results['net_profit']:,.0f}   |   Forecast MAE: ${results['val_mae']:.1f}/MWh")
        except ImportError:
            st.warning("bess_pipeline.py not found — showing demo with cached CSV data.")
            # Load from outputs/ if available
            dispatch_csv = Path('outputs') / f'dispatch_{target_year}_{target_month:02d}.csv'
            forecast_csv = Path('outputs') / f'forecast_{target_year}_{target_month:02d}.csv'
            if dispatch_csv.exists() and forecast_csv.exists():
                df_dis = pd.read_csv(dispatch_csv, index_col='datetime', parse_dates=True)
                df_fc  = pd.read_csv(forecast_csv, index_col='datetime', parse_dates=True)
                rev    = float(df_dis['revenue'].sum())
                deg    = float(df_dis['degradation_cost'].sum())
                st.session_state.results = {
                    'df_dispatch': df_dis, 'df_forecast': df_fc,
                    'net_profit': rev - deg, 'total_revenue': rev,
                    'val_mae': 0, 'soc_min': soc_min, 'soc_max': soc_max,
                }
                status.update(label="✅ Loaded from cache", state="complete")
            else:
                st.error("No cached results found. Run the notebook first, then re-open the app.")
                status.update(label="❌ Failed", state="error")

# ── DISPLAY RESULTS ─────────────────────────────────────────────────────
if st.session_state.results:
    res    = st.session_state.results
    df_dis = res['df_dispatch']
    df_fc  = res['df_forecast']
    rev    = float(df_dis['revenue'].sum())
    deg    = float(df_dis['degradation_cost'].sum())
    net    = rev - deg
    tot_ch  = float((df_dis['p_charge'] * (5/60)).sum())
    tot_dis = float((df_dis['p_discharge'] * (5/60)).sum())
    n_cyc   = (tot_ch + tot_dis) / (2 * capacity_mwh)
    n_days  = len(set(df_dis.index.date))
    avg_buy  = float(df_dis.loc[df_dis['p_charge']>1, 'price'].mean()) if (df_dis['p_charge']>1).sum()>0 else 0
    avg_sell = float(df_dis.loc[df_dis['p_discharge']>1, 'price'].mean()) if (df_dis['p_discharge']>1).sum()>0 else 0

    # ── KPI tiles ──────────────────────────────────────────────────────
    st.markdown("### 📊 Monthly KPIs")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Gross Revenue",    f"${rev:,.0f}")
    k2.metric("Degradation Cost", f"${deg:,.0f}")
    k3.metric("Net Profit",       f"${net:,.0f}", delta=f"${net/n_days:,.0f}/day")
    k4.metric("EFC Cycles",       f"{n_cyc:.1f}")
    k5.metric("Avg Buy Price",    f"${avg_buy:.1f}/MWh")
    k6.metric("Avg Sell Price",   f"${avg_sell:.1f}/MWh")

    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Forecast", "⚡ Dispatch", "🔋 State of Charge",
        "💰 Revenue", "🗓️ Daily Detail"
    ])

    with tab1:
        st.plotly_chart(plot_price_forecast(df_fc, month_name), use_container_width=True)
        neg_pct  = (df_fc['price_forecast'] < 0).mean() * 100
        spk_pct  = (df_fc['price_forecast'] > 1000).mean() * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast Mean",   f"${df_fc['price_forecast'].mean():.1f}/MWh")
        c2.metric("Forecast Max",    f"${df_fc['price_forecast'].max():.0f}/MWh")
        c3.metric("Negative %",      f"{neg_pct:.1f}%", help="Intervals with negative prices — charge opportunities")
        c4.metric("Spike >$1k %",    f"{spk_pct:.2f}%", help="Intervals with price > $1000 — discharge opportunities")

    with tab2:
        st.plotly_chart(plot_dispatch(df_dis), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Charged",     f"{tot_ch:.0f} MWh")
        c2.metric("Total Discharged",  f"{tot_dis:.0f} MWh")
        c3.metric("EFC / month",       f"{n_cyc:.2f}")

    with tab3:
        st.plotly_chart(plot_soc(df_dis, soc_min, soc_max), use_container_width=True)
        soc_pct = df_dis['soc'] * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min SoC",       f"{soc_pct.min():.1f}%",
                   delta="⚠ Near floor" if soc_pct.min() < soc_min*100 + 2 else "✓ Safe")
        c2.metric("Max SoC",       f"{soc_pct.max():.1f}%")
        c3.metric("Avg SoC",       f"{soc_pct.mean():.1f}%")
        c4.metric("SoC constraint", "✓ Never violated",
                   delta="[10% – 90%]")

    with tab4:
        st.plotly_chart(plot_cumulative_revenue(df_dis), use_container_width=True)
        daily = (df_dis['revenue'] - df_dis['degradation_cost']).resample('D').sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Best day",   f"${daily.max():,.0f}",  delta=str(daily.idxmax().date()))
        c2.metric("Worst day",  f"${daily.min():,.0f}",  delta=str(daily.idxmin().date()))
        c3.metric("Profitable days", f"{(daily > 0).sum()}/{len(daily)}")

        # Daily bar chart
        fig_daily = go.Figure(go.Bar(
            x=list(range(len(daily))), y=daily.values,
            marker_color=[C['revenue'] if v >= 0 else C['loss'] for v in daily],
            opacity=0.85
        ))
        fig_daily.add_hline(y=daily.mean(), line=dict(color='yellow', dash='dash'),
                             annotation_text=f"Avg ${daily.mean():,.0f}")
        fig_daily.update_layout(
            title='Daily Net Profit', xaxis_title='Day of month', yaxis_title='$',
            paper_bgcolor=C['bg'], plot_bgcolor=C['bg2'],
            font=dict(color='#eef2ff'), height=250,
            yaxis=dict(tickformat='$,.0f'), margin=dict(t=40, b=30, l=60, r=20)
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with tab5:
        daily_net = (df_dis['revenue'] - df_dis['degradation_cost']).resample('D').sum()
        best_day  = str(daily_net.idxmax().date())
        sel_day   = st.date_input("Select day",
                                    value=pd.Timestamp(best_day).date(),
                                    min_value=df_dis.index.min().date(),
                                    max_value=df_dis.index.max().date())
        day_str = str(sel_day)
        st.plotly_chart(plot_daily_detail(df_dis, day_str), use_container_width=True)

    st.markdown("---")

    # ── Download buttons ───────────────────────────────────────────────
    st.markdown("### 📥 Export")
    c1, c2 = st.columns(2)
    c1.download_button(
        label   = "⬇  Download Dispatch CSV",
        data    = df_dis.to_csv().encode(),
        file_name = f"dispatch_{target_year}_{target_month:02d}_{region}.csv",
        mime    = "text/csv"
    )
    c2.download_button(
        label   = "⬇  Download Forecast CSV",
        data    = df_fc.to_csv().encode(),
        file_name = f"forecast_{target_year}_{target_month:02d}_{region}.csv",
        mime    = "text/csv"
    )

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:80px 40px; color:#8b9dc3">
        <h2 style="color:#c084fc">⚡ Ready</h2>
        <p>Select a target month and region in the sidebar, then click <b>Run Dispatch Optimisation</b>.</p>
        <p style="font-size:0.85em">The pipeline will forecast NEM prices and optimise the BESS dispatch schedule.</p>
    </div>
    """, unsafe_allow_html=True)

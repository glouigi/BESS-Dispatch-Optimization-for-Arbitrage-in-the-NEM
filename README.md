# ⚡ BESS Dispatch Optimization for Arbitrage in the NEM

**ML Price Forecasting → Battery Storage Optimal Dispatch across all NEM regions**

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white&link=https://github.com/glouigi)](https://github.com/glouigi)
[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white&link=www.linkedin.com/in/giorgio-louigi-ramirez-quiroz-924a2872)](https://www.linkedin.com/in/giorgio-louigi-ramirez-quiroz-924a2872/)
[![Gmail Badge](https://img.shields.io/badge/-Gmail-c14438?style=for-the-badge&logo=Gmail&logoColor=white&link=mailto:contato.weltonf@gmail.com)](mailto:g.ramirezqui@gmail.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](BESS_Optimized_Dispatch.ipynb)
[![Streamlit](https://img.shields.io/badge/Streamlit-Control_Room_App-ff4b4b?logo=streamlit)](app.py)
[![AEMO](https://img.shields.io/badge/Data-AEMO_TRADINGPRICE-0078d4)](https://nemweb.com.au)
[![Framework](https://img.shields.io/badge/Master_ML%2FDL-12_Steps-c084fc)](BESS_Optimized_Dispatch.ipynb)

---

## 📋 Project Description

End-to-end machine learning pipeline for optimising the daily charge/discharge schedule
of a utility-scale Battery Energy Storage System (BESS) operating in the **Australian
National Electricity Market (NEM)** — covering SA1, VIC1, NSW1, QLD1, and TAS1.

### Pipeline Overview

```
AEMO TRADINGPRICE (30-min, 2 years)
         │
         ▼  STEP 2 — Data: NEMOSIS download, integrity checks, caching
         │
         ▼  STEP 3 — EDA: duck curve, autocorrelation, price regime analysis
         │
         ▼  STEP 4 — Validation: temporal split, test set locked
         │
         ▼  STEPS 5–6 — Features: 80+ features (lags, rolling stats,
         │              cyclical encoding, negative-price regime flags)
         │              + RobustScaler pipeline
         │
         ▼  STEP 7 — Baselines: seasonal naive, rolling mean/median
         │
         ▼  STEPS 8–9 — ML: XGBoost + LightGBM ensemble
         │              Raw price target (negatives fully visible)
         │              Optuna TPE hyperparameter optimisation (optional)
         │
         ▼  STEP 10 — Evaluation: SHAP importance, residuals, one-time test set
         │
         ▼  STEP 11 — Forecast: Seasonal Anchor method
         │             (preserves duck curve + negative prices over full month)
         │
         ▼  STEP 11b — LP/MILP Dispatch: CVXPY + GLPK_MI
         │              Multi-cycle, negative-price priority charging
         │
         ▼  STEP 12 — Dashboard: price, dispatch, SoC, revenue, KPIs
         │
         ▼  Post-month KPI evaluation: true MAE vs real AEMO prices
```

### Why This Approach

| Design Decision | Reason |
|----------------|--------|
| **30-min TRADINGPRICE** | Day-ahead scheduling interval — what AEMO NEMDE uses |
| **Raw price target (not log)** | Log transform crushes negative price signal — model learns to ignore negatives |
| **Seasonal Anchor forecast** | Prevents recursive collapse to flat mean over monthly horizon |
| **LP/MILP dispatch** | Globally optimal for day-ahead with known forecast — faster and more interpretable than RL |
| **Negative-price priority** | NEM negative prices = free/paid charging — highest-value opportunity |

---

## 🌏 NEM Regions Supported

| Region | State | Characteristics |
|--------|-------|-----------------|
| **SA1** | South Australia | Strongest duck curve · Highest negative price rate · Most volatile |
| **VIC1** | Victoria | Strong solar · Evening ramp · Interconnector dynamics |
| **NSW1** | New South Wales | Largest load · Moderate volatility · Coal transition |
| **QLD1** | Queensland | Afternoon peaks · Rooftop solar growing rapidly |
| **TAS1** | Tasmania | Hydro-dominated · Low volatility · Basslink interconnector |

---

## 🔋 BESS Asset Configuration

Designed for **any utility-scale BESS** connected to the NEM. Configured in notebook cell 1:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Technology | LFP | — | Lithium Iron Phosphate |
| Power | 100 MW | 10–500 MW | Rated charge/discharge power |
| Energy | 200 MWh | 20–2000 MWh | Nominal capacity |
| Duration | 2 h | 0.5–8 h | E/P ratio |
| Round-trip eff. | 90.25% | — | η_c × η_d = 0.95² |
| SoC window | 10–90% | 5–95% | Usable SoC range |
| Cycle life | 4,000 EFC | — | LFP at 80% DoD |
| CAPEX | $220/kWh | — | → $44M for 200 MWh |
| Cycle cost | ~$5.5/MWh | — | Marginal degradation |
| Break-even spread | ~$11.6/MWh | — | Min spread to profit |

---

## 📈 Expected Results

| Metric | Typical Value | KPI Target |
|--------|--------------|------------|
| Forecast MAE | $18–26/MWh | < $25/MWh |
| vs seasonal naive | −35–45% MAE | Beat baseline |
| Negative price detection | Visible in forecast | > 0% |
| Net profit (SA1, 100MW/200MWh) | $100k–280k/month | > $0 |
| Dispatch efficiency | 84–92% | ≥ 85% of perfect-foresight |
| EFC cycles/month | 14–28 | — |

---

## 🚀 Quick Start

```bash
# 1 — Clone
git clone https://github.com/YOUR_USERNAME/bess-optimized-dispatch.git
cd bess-optimized-dispatch

# 2 — Install
pip install -r requirements.txt

# 3 — Run notebook
jupyter lab BESS_Optimized_Dispatch.ipynb
```

In cell 1, set your target:
```python
TARGET_YEAR   = 2024    # ← any year
TARGET_MONTH  = 6       # ← 1=Jan … 12=Dec
NEM_REGION    = 'SA1'   # ← SA1 | VIC1 | NSW1 | QLD1 | TAS1
```
`Kernel → Restart & Run All`

---

## 🔄 Re-forecast and Daily Dispatch

```python
# Re-forecast any month (models stay in memory, ~1-2 min):
r = quick_forecast(2025, 3)                         # March 2025
r = quick_forecast(2025, 6, plot_date='2025-06-28') # with specific day

# Plot any day (month-end dates work correctly):
plot_day_dispatch(df_dis, df_fc, date='2024-06-30') # June 30 ✓
plot_day_dispatch(df_dis, df_fc, date='2024-02-28') # Feb 28 ✓
plot_day_dispatch(df_dis, df_fc, date='2024-02-29') # Feb 29 leap year ✓

# Post-month true KPI evaluation:
kpi = evaluate_post_month(df_fc, df_dis, 2024, 6)
```

---

## 🌐 Control Room Web App

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
# All machines on local network: http://SERVER-IP:8501
```

See **STREAMLIT_DEPLOY.md** for full deployment options:
- Option A: Local network server (control room)
- Option B: Streamlit Community Cloud (free, internet)
- Option C: Docker (AWS / Azure / GCP)

---

## 📁 Repository Structure

```
bess-optimized-dispatch/
│
├── BESS_Optimized_Dispatch.ipynb   ← Main ML pipeline (all 12 steps)
├── app.py                           ← Streamlit control room app
├── bess_pipeline.py                 ← Pipeline functions for Streamlit
│
├── data/                            ← NEMOSIS cache (auto-created)
├── outputs/                         ← Plots + CSV exports
├── models/                          ← Saved models (after running notebook)
│
├── STREAMLIT_DEPLOY.md              ← Deployment guide (3 options)
├── GITHUB_SETUP.md                  ← GitHub push guide
├── requirements.txt
├── environment.yml
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` `pandas` `scipy` | Core |
| `scikit-learn` | Pipeline, scaler, metrics |
| `xgboost` `lightgbm` | Forecast models |
| `cvxpy` | LP/MILP dispatch solver |
| `optuna` | HPO (optional) |
| `shap` | Interpretability (optional) |
| `nemosis` | Real AEMO data download |
| `matplotlib` | Notebook plots |
| `plotly` `streamlit` | Web app |

---

## 📡 Data Source

- **AEMO NEMWEB**: https://nemweb.com.au
- **Table**: TRADINGPRICE — 30-min Regional Reference Price (RRP)
- **Library**: NEMOSIS (open-source, pip install nemosis)
- **License**: CC BY 4.0 — free to use with attribution

---


## 📚 References

1. AEMO. *National Electricity Market Data Model*. 2024.
2. Chen & Guestrin. XGBoost. *KDD* 2016.
3. Ke et al. LightGBM. *NeurIPS* 2017.
4. Akiba et al. Optuna. *KDD* 2019.
5. Lundberg & Lee. SHAP. *NeurIPS* 2017.

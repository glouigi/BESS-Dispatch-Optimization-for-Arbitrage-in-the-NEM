# ⚡ BESS Control Room — Streamlit Deployment Guide

Complete step-by-step instructions to deploy the BESS Optimized Dispatch
as a web application accessible by your entire operations team.

---

## What You Are Deploying

A browser-based control room dashboard where operators can:
- Select any NEM region and target month
- Run price forecast + BESS dispatch optimisation
- View interactive charts: Price · Dispatch · SoC · Revenue
- Plot any specific day in detail
- Download dispatch CSV results

No Jupyter required. Works on any browser.

---

## OPTION A — Local Network (Control Room Server)

**Best for**: Internal company use, no internet exposure, quick setup.

### Step 1 — Install dependencies

```bash
# On the server that will run the app:
pip install -r requirements.txt
pip install streamlit plotly
```

### Step 2 — Run the notebook once first (trains the ML models)

```bash
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 \
    BESS_Optimized_Dispatch.ipynb
```

This saves `models/pipeline.pkl`, `models/xgb.pkl`, `models/lgbm.pkl`.
The Streamlit app loads these — you only train once.

### Step 3 — Launch the app

```bash
# Accessible only on this machine:
streamlit run app.py

# Accessible to ALL machines on the same network:
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Step 4 — Access from any browser

```
http://SERVER-IP-ADDRESS:8501
```

Find your server IP:
```bash
# Linux/Mac:
ip addr show | grep "inet " | grep -v 127.0.0.1

# Windows:
ipconfig | findstr "IPv4"
```

### Step 5 — Keep it running (Linux systemd service)

Create `/etc/systemd/system/bess-dispatch.service`:

```ini
[Unit]
Description=BESS Optimized Dispatch — Streamlit App
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/bess-optimized-dispatch
ExecStart=/usr/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable bess-dispatch
sudo systemctl start bess-dispatch
sudo systemctl status bess-dispatch    # verify running
```

App now starts automatically on server boot.

---

## OPTION B — Streamlit Community Cloud (Free, Public/Private)

**Best for**: Access from anywhere, no server to manage.

### Step 1 — Push code to GitHub

```bash
git init
git add BESS_Optimized_Dispatch.ipynb app.py bess_pipeline.py \
        requirements.txt .gitignore README.md
git commit -m "feat: BESS dispatch app"
git remote add origin https://github.com/YOUR_USERNAME/bess-optimized-dispatch.git
git push -u origin main
```

### Step 2 — Sign up at share.streamlit.io

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **New app**

### Step 3 — Configure deployment

```
Repository:    YOUR_USERNAME/bess-optimized-dispatch
Branch:        main
Main file:     app.py
```

Click **Deploy** — Streamlit installs dependencies and launches.

### Step 4 — Set secrets (NEMOSIS credentials if needed)

In Streamlit Cloud → App settings → **Secrets**:

```toml
[nemosis]
# Add any API keys here if your NEMOSIS setup requires them
# NEMOSIS is open access — usually no keys needed
```

### Step 5 — Share the URL

Streamlit gives you a URL like:
`https://your-username-bess-dispatch-app-xyz.streamlit.app`

Share with your team. Set the app to **Private** in settings to restrict access.

---

## OPTION C — Docker (Enterprise, any cloud)

**Best for**: AWS / Azure / GCP / on-premise Docker hosts.

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install streamlit plotly

COPY . .
RUN mkdir -p data outputs models

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
```

### Build and run

```bash
# Build:
docker build -t bess-dispatch .

# Run locally:
docker run -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/outputs:/app/outputs \
    bess-dispatch

# Access at: http://localhost:8501
```

### Push to cloud (example: AWS ECR)

```bash
# Authenticate
aws ecr get-login-password --region ap-southeast-2 | \
    docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT.dkr.ecr.ap-southeast-2.amazonaws.com

# Tag and push
docker tag bess-dispatch:latest YOUR_AWS_ACCOUNT.dkr.ecr.ap-southeast-2.amazonaws.com/bess-dispatch:latest
docker push YOUR_AWS_ACCOUNT.dkr.ecr.ap-southeast-2.amazonaws.com/bess-dispatch:latest

# Deploy on ECS / App Runner / EC2 — see AWS docs
```

---

## Streamlit Configuration File

Create `.streamlit/config.toml` in your project root:

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
base = "dark"
primaryColor = "#c084fc"
backgroundColor = "#07090f"
secondaryBackgroundColor = "#0d1117"
textColor = "#eef2ff"
font = "monospace"
```

---

## Troubleshooting

**App shows "models not found"**
→ Run the notebook first: `Kernel → Restart & Run All`
→ Confirm `models/pipeline.pkl` and `models/xgb.pkl` exist

**NEMOSIS download fails in app**
→ Ensure internet access from the server
→ Check `data/raw_nemosis/` has write permissions
→ NEMOSIS is free and open — no API key needed

**App is slow on first forecast**
→ Normal — NEMOSIS downloads ~24 monthly CSV files on first run
→ Subsequent runs use the parquet cache in `data/` (seconds)

**Port 8501 blocked**
```bash
# Check firewall:
sudo ufw status                          # Ubuntu
sudo firewall-cmd --list-ports           # CentOS/RHEL
# Open port:
sudo ufw allow 8501/tcp
```

**Streamlit version conflict**
```bash
pip install streamlit==1.35.0 --force-reinstall
```

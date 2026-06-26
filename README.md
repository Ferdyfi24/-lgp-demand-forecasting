📈 LPG Demand Forecasting Analysis (Dec 2024 – May 2026)
> **Python-based demand forecasting project comparing SMA, WMA, and SES methods to predict monthly LPG cylinder demand across 3 coverage areas of a Pertamina-authorized sub-agent.**
🎯 Objective
Identify the most accurate forecasting method for monthly LPG demand planning, enabling better stock allocation and reorder timing decisions across Area A (Urban), Area B (Semi-urban), and Area C (Rural).
🛠️ Tools
`Python` · `Pandas` · `NumPy` · `Matplotlib`
🔍 Methods Compared
Method	Description
SMA-3	Simple Moving Average (3-month window)
WMA-3	Weighted Moving Average (0.5 / 0.3 / 0.2)
SES	Simple Exponential Smoothing (α=0.4)
📊 Visualizations
Actual vs all forecast methods + 3-month forward projection (Jun–Aug 2026)
Area-level demand trend with smoothing overlays
MAE & MAPE accuracy comparison bar chart
📁 Structure
```
lgp-demand-forecasting/
├── demand_forecasting.py
├── README.md
└── outputs/
    ├── plot_forecast_comparison.png
    ├── plot_area_forecast.png
    └── plot_forecast_error.png
```
🚀 How to Run
```bash
# Google Colab: paste cell by cell and run
# Local:
pip install pandas numpy matplotlib
python demand_forecasting.py
```
📌 CV Bullet (Ops/Logistics CV — Analytical Projects)
> **LPG Demand Forecasting Analysis** | Python, Pandas, NumPy | Dec 2024 – May 2026
> Developed and compared 3 demand forecasting models (SMA, WMA, SES) on 18-month LPG distribution data across 3 coverage areas (~2,000 cylinders/month); evaluated model accuracy using MAE and MAPE metrics to identify optimal method for monthly stock allocation and reorder planning.
👤 Author
Ferdy Febrian Iskandar · iskandarferdy559@gmail.com
Dataset simulated based on real operational experience. For portfolio purposes only.

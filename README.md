# AI Quant Portfolio: LSTM vs Transformer

This project implements deep learning models (LSTM and Transformer) for **multi-asset return forecasting** and **dynamic portfolio allocation**.  
It replicates the methodology from [Artificial Intelligence in Quantitative Finance, 2021].

---

## 📂 Project Structure
ai-quant-portfolio/
├── src/                     # all Python scripts
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   ├── optimizer_layer.py
│   ├── backtest.py
│   └── utils.py
├── main.py                  # driver script
├── requirements.txt         # dependencies
├── README.md                # clear instructions
├── .gitignore               # ignore junk
├── results/                 # metrics + plots (after run)
│   ├── nav_comparison.png
│   ├── metrics_comparison.json
│   ├── nav_transformer.csv
│   └── nav_lstm.csv
└── report/
    ├── explanation_report.md
    └── explanation_report.pdf




---

## ⚡ Quickstart

1. Clone the repo:
   ```bash
   git clone https://github.com/shivamsinghgaur291/ai-quant-portfolio.git
   cd ai-quant-portfolio

2. Install dependencies:
             pip install -r requirements.txt


3. Run experiments:
             python main.py


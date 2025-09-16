# main.py
import os
import json
import torch
import matplotlib.pyplot as plt

from src.data_loader import download_data, compute_log_returns
from src.features import rolling_features
from src.train import fit_pipeline
from src.backtest import simulate_daily
from src.utils import metrics_from_nav


def run_experiment(model_type="transformer"):
    print(f"\n=== Running {model_type.upper()} Experiment ===")
    # 1. Download prices
    prices = download_data(start="2010-01-01", end="2020-12-31", save_csv=True)
    rets = compute_log_returns(prices)

    # 2. Build features
    feat_df = rolling_features(prices, rets)

    # 3. Train model
    model, (X_test, y_test) = fit_pipeline(
        X_df=feat_df,
        y_df=rets.loc[feat_df.index],
        lookback=60,
        model_type=model_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 4. Backtest
    nav, weights = simulate_daily(
        prices.loc[feat_df.index],
        model,
        feat_df,
        scaler=None,
        lookback=60,
        tx_cost=0.001,
    )

    # ensure results dir
    os.makedirs("results", exist_ok=True)

    # save NAV to CSV
    nav.to_csv(f"results/nav_{model_type}.csv")

    # 5. Metrics
    metrics = metrics_from_nav(nav)
    print(f"{model_type.upper()} metrics:", metrics)

    # save metrics to JSON
    with open(f"results/metrics_{model_type}.json", "w") as f:
        json.dump(metrics, f, indent=4, default=float)

    return nav, metrics


if __name__ == "__main__":
    # Run both experiments
    nav_transformer, transformer_results = run_experiment("transformer")
    nav_lstm, lstm_results = run_experiment("lstm")

    print("\n=== Comparison ===")
    print("Transformer:", transformer_results)
    print("LSTM:", lstm_results)

    # Save combined metrics
    with open("results/metrics_comparison.json", "w") as f:
        json.dump(
            {"transformer": transformer_results, "lstm": lstm_results},
            f,
            indent=4,
            default=float,
        )

    # Plot NAV comparison
    plt.figure(figsize=(12, 6))
    nav_transformer.plot(label="Transformer")
    nav_lstm.plot(label="LSTM")
    plt.title("NAV Comparison")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/nav_comparison.png")
    plt.show()

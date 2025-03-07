import os
import json
import argparse
import matplotlib.pyplot as plt

def load_results(save_dir, ratios):
    results = {}
    for ratio in ratios:
        history_path = os.path.join(save_dir, f"training_history_{ratio}.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                results[ratio] = json.load(f)
        else:
            print(f"Warning: No training history found for {ratio}% synthetic data.")
    return results

def plot_results(results, metric, output_dir):
    plt.figure(figsize=(10, 6))

    for ratio, data in results.items():
        if "metrics" in data and metric in data["metrics"]:
            plt.plot(data["metrics"][metric], label=f"{ratio}% synthetic")
        else:
            print(f"Warning: Metric {metric} not found for {ratio}% synthetic data.")

    plt.xlabel("Epochs")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Comparison of {metric} across different synthetic ratios")
    plt.legend()
    plt.grid(True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save plot to disk
    plot_path = os.path.join(output_dir, f"{metric.replace('/', '_')}_comparison.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training results for different synthetic ratios")
    parser.add_argument("--save_dir", type=str, default="runs/train", help="Directory containing training results")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--ratios", type=int, nargs="+", default=[0, 5, 10, 15, 20], help="List of synthetic ratios to compare")
    parser.add_argument("--metric", type=str, default="train/loss", help="Metric to plot (e.g., train/loss, val/accuracy)")

    args = parser.parse_args()

    results = load_results(args.save_dir, args.ratios)
    if results:
        plot_results(results, args.metric, args.output_dir)
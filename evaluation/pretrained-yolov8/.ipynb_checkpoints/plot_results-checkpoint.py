import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_results(save_dir, ratios):
    results = {}
    for ratio in ratios:
        csv_path = os.path.join(save_dir, f"training_history_{ratio}.csv")
        if os.path.exists(csv_path):
            results[ratio] = pd.read_csv(csv_path)
        else:
            print(f"Warning: No training history found for {ratio}% synthetic data.")
    return results

def extract_map_at_epoch(results, epoch=50):
    map_results = {}
    for ratio, data in results.items():
        if "metrics/mAP50(B)" in data.columns:
            if len(data) >= epoch:
                map_results[ratio] = data["metrics/mAP50(B)"].iloc[epoch - 1]  # 50th epoch is indexed at 49
            else:
                print(f"Warning: Epoch {epoch} data not available for {ratio}% synthetic data.")
        else:
            print(f"Warning: mAP metric not found for {ratio}% synthetic data.")
    return map_results

def plot_results(map_results, output_dir):
    plt.figure(figsize=(10, 6))

    # Plotting the mAP values for available ratios
    for ratio, mAP in map_results.items():
        plt.plot(ratio, mAP, 'bo', label=f"{ratio}% synthetic" if ratio != 0 else "0% synthetic")

    plt.xlabel("Synthetic Data Ratio (%)")
    plt.ylabel("mAP50 (B)")
    plt.title("Comparison of mAP50 (B) at Epoch 50 for Different Synthetic Ratios")
    plt.legend()
    plt.grid(True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save plot to disk
    plot_path = os.path.join(output_dir, "mAP50_epoch50_comparison.png")
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mAP at epoch 50 for different synthetic ratios")
    parser.add_argument("--save_dir", type=str, default="runs/train", help="Directory containing training results")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--ratios", type=int, nargs="+", default=[0, 5, 10, 15, 20], help="List of synthetic ratios to compare")

    args = parser.parse_args()

    results = load_results(args.save_dir, args.ratios)
    if results:
        map_results = extract_map_at_epoch(results, epoch=50)
        if map_results:
            plot_results(map_results, args.output_dir)
        else:
            print("No mAP results found for the specified epoch.")

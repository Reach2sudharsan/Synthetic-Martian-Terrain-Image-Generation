import yaml
import random
import os

# Paths to original dataset YAMLs
real_yaml_path = "../Synthetic-Martian-Terrain-Image-Generation/datasets/Mars-Lunar-Crater-yolov8/data.yaml"
synthetic_yaml_path = "../Synthetic-Martian-Terrain-Image-Generation/datasets/SyntheticCraters/data.yaml"
output_yaml_dir = "../Synthetic-Martian-Terrain-Image-Generation/datasets/new_datasets"  # Folder to store generated YAML files
os.makedirs(output_yaml_dir, exist_ok=True)

# Load original YAML files
with open(real_yaml_path, 'r') as file:
    real_yaml = yaml.safe_load(file)

with open(synthetic_yaml_path, 'r') as file:
    synthetic_yaml = yaml.safe_load(file)

# Extract train and validation image paths
real_train_images = real_yaml['train']  # List of real training images
real_val_images = real_yaml['val']  # Validation images remain unchanged
synthetic_train_images = synthetic_yaml['train']  # List of synthetic images

# Keep validation set unchanged
val_images = real_val_images

# Number of original training images
N_train_original = len(real_train_images)

# Define synthetic-to-real ratios to test
ratios = [0, 5, 10, 15, 20]  # Percentages

for ratio in ratios:
    N_synthetic = int(N_train_original * (ratio / 100))  # How many synthetic images to add
    N_real = N_train_original - N_synthetic  # How many real images to keep

    # Select synthetic images (ensure we don't exceed available)
    selected_synthetic = random.sample(synthetic_train_images, min(N_synthetic, len(synthetic_train_images)))

    # Select real images (random subset to maintain total size)
    selected_real = random.sample(real_train_images, N_real)

    # Combine real and synthetic images
    combined_train = selected_real + selected_synthetic

    # Create new YAML structure
    new_yaml = {
        "train": combined_train,
        "val": val_images,  # Keep validation set unchanged
        "nc": real_yaml["nc"],  # Number of classes
        "names": real_yaml["names"]  # Class names
    }

    # Save new YAML file
    yaml_filename = f"{output_yaml_dir}/crater_dataset_{ratio}synthetic.yaml"
    with open(yaml_filename, "w") as file:
        yaml.dump(new_yaml, file)

    print(f"Generated: {yaml_filename} (Real: {N_real}, Synthetic: {N_synthetic}, Total: {len(combined_train)})")

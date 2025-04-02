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

# Extract train and validation image directories (paths to the directories, not file lists)
real_train_dir = os.path.abspath("../Synthetic-Martian-Terrain-Image-Generation/datasets/Mars-Lunar-Crater-yolov8/train/images")  # Absolute path to real training images folder
real_val_dir = os.path.abspath("../Synthetic-Martian-Terrain-Image-Generation/datasets/Mars-Lunar-Crater-yolov8/valid/images")  # Absolute path to validation images folder
synthetic_train_dir = os.path.abspath("../Synthetic-Martian-Terrain-Image-Generation/datasets/SyntheticCraters/train/images")  # Absolute path to synthetic images folder

# Print paths to check if they are correct
print(f"Real train directory: {real_train_dir}")
print(f"Synthetic train directory: {synthetic_train_dir}")
print(f"Real validation directory: {real_val_dir}")

# Check if directories exist
if not os.path.isdir(real_train_dir):
    print(f"Error: {real_train_dir} does not exist!")
if not os.path.isdir(synthetic_train_dir):
    print(f"Error: {synthetic_train_dir} does not exist!")
if not os.path.isdir(real_val_dir):
    print(f"Error: {real_val_dir} does not exist!")

# Number of original training images
N_train_original = len(os.listdir(real_train_dir)) if os.path.isdir(real_train_dir) else 0

# Define synthetic-to-real ratios to test
ratios = [0, 5, 10, 15, 20]  # Percentages

for ratio in ratios:
    N_synthetic = int(N_train_original * (ratio / 100))  # How many synthetic images to add
    N_real = N_train_original - N_synthetic  # How many real images to keep

    # Select synthetic images (ensure we don't exceed available)
    synthetic_images = os.listdir(synthetic_train_dir) if os.path.isdir(synthetic_train_dir) else []
    selected_synthetic = random.sample(synthetic_images, min(N_synthetic, len(synthetic_images)))

    # Select real images (random subset to maintain total size)
    real_images = os.listdir(real_train_dir) if os.path.isdir(real_train_dir) else []
    selected_real = random.sample(real_images, N_real)

    # Combine real and synthetic images
    combined_train = selected_real + selected_synthetic

    # Create new YAML structure with absolute paths
    new_yaml = {
        "train": real_train_dir,  # Absolute path to the directory where images are stored
        "val": real_val_dir,  # Absolute path for validation set
        "nc": real_yaml["nc"],  # Number of classes
        "names": real_yaml["names"]  # Class names
    }

    # Save new YAML file
    yaml_filename = f"{output_yaml_dir}/crater_dataset_{ratio}synthetic.yaml"
    with open(yaml_filename, "w") as file:
        yaml.dump(new_yaml, file)

    print(f"Generated: {yaml_filename} (Real: {N_real}, Synthetic: {N_synthetic}, Total: {len(combined_train)})")
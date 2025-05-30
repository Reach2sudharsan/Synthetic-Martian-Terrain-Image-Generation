#!/bin/bash
#SBATCH - N1                                      # Request 1 Node
#SBATCH --job-name=train_detector
#SBATCH -p GPU-shared                             # Use GPU-shared partition
#SBATCH -t 1:20:00                                # Job time limit: 1hr 20min
#SBATCH --gres=gpu:1                              # Request 1 GPU
#SBATCH -A eng240004p
#SBATCH --mail-user=sgopalakrishnan@g.hmc.edu     # Email for job updates
#SBATCH --mail-type=END,FAIL                      # Email on job completion or failure
#SBATCH --output=train_detector_output.txt        # Store output in text file

export CTRLNET=/jet/home/sgopala2/Synthetic-Martian-Terrain-Image-Generation

source source /jet/home/sgopala2/miniconda3/sgauton/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    exit 1
fi

echo "Python 3 executable: $(which python3)"
echo "Python 3 version: $(python3 --version)"

echo "Checking for GPU:"
nvidia-smi || echo "nvidia-smi failed — no GPU visible"

# Run program
echo "Running train_detector.py..."
python3 /jet/home/sgopala2/Synthetic-Martian-Terrain-Image-Generation/train_detector.py

import cv2
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
import random

# Define the remote input file and output base name
input_file = "/home/ubuntu/dem-controlnet/Synthetic-Martian-Terrain-Image-Generation/Large_Files/JEZ_MADNet_DTM_50cm_v8.tif"

with rasterio.open(input_file) as src:
    width, height = src.width, src.height
    crs = src.crs  # Coordinate reference system
    transform = src.transform  # Geospatial transform
    print(f"Width: {width} pixels, Height: {height} pixels")
    print(f"CRS: {crs}")
    print(f"Transform: {transform}")

# Define output folder
output_folder = "DEM_Images"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

# Crop size
crop_width, crop_height = 2000, 2000  

# Define middle 20000x20000 region
mid_x_start = (41935 - 20000) // 2  # Start x of middle square
mid_y_start = (42847 - 20000) // 2  # Start y of middle square
mid_x_end = mid_x_start + 20000 - crop_width  # Max x within middle square
mid_y_end = mid_y_start + 20000 - crop_height  # Max y within middle square

count = 100
# Generate 200 random (x, y) positions within the middle region
random_positions = [(random.randint(mid_x_start, mid_x_end), random.randint(mid_y_start, mid_y_end)) for _ in range(count)]

# Open the DEM file
with rasterio.open(input_file) as src:
    for i, (x_offset, y_offset) in enumerate(tqdm(random_positions, desc="Extracting Random DEM Crops", unit="crop")):
        window = Window(x_offset, y_offset, crop_width, crop_height)

        # Read the cropped DEM data
        dem_data = src.read(1, window=window)

        # Normalize DEM values for 8-bit PNG
        dem_min, dem_max = np.min(dem_data), np.max(dem_data)
        dem_normalized = ((dem_data - dem_min) / (dem_max - dem_min) * 255).astype(np.uint8)

        # Save as PNG in the "DEM_Images" folder
        output_file = os.path.join(output_folder, f"mars_dem_random_{i+1}.png")
        cv2.imwrite(output_file, dem_normalized)

        tqdm.write(f"Saved cropped DEM as PNG: {output_file} (Offset: x={x_offset}, y={y_offset})")

print(f"✅ All {count} DEM crops saved in '{output_folder}' successfully!")
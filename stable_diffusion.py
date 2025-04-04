import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
from tqdm import tqdm


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. PyTorch will use the GPU.')
    print('Device name:', torch.cuda.get_device_name(0)) # Get GPU name
    print('Number of GPUs available:', torch.cuda.device_count())
else:
    device = torch.device('cpu')
    print('CUDA is not available. PyTorch will use the CPU.')


# Load Stable Diffusion (without conditioning)
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# Load ControlNet Model
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to("cuda")
pipeline_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
).to("cuda")

def generate_diffusion_image(prompt, image):
    # Convert single-channel grayscale to 3-channel format
    image_rgb = np.stack([image] * 3, axis=-1)  # Replicate across RGB channels

    # Convert to tensor
    image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)

    # Generate image using ControlNet
    image = pipeline(prompt, image=image_tensor).images[0]
    return image
    
def generate_controlnet_image(prompt, image):
    # Convert single-channel grayscale to 3-channel format
    image_rgb = np.stack([image] * 3, axis=-1)  # Replicate across RGB channels

    # Convert to tensor
    image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)

    # Generate image using ControlNet
    image = pipeline_controlnet(prompt, image=image_tensor).images[0]
    return image

# Ensure StableDiffusion image output folder exists
stable_diffusion_output_dir = "stable_diffusion_images"
os.makedirs(stable_diffusion_output_dir, exist_ok=True)

# Get all DEM images
dem_folder = "../dem-controlnet/DEM_Images"
dem_images = [f for f in os.listdir(dem_folder) if f.endswith(".png")]

# Process each DEM image through StableDiffusion with tqdm progress bar
for i, dem_image in enumerate(tqdm(dem_images, desc="Generating Stable Diffusion Images", unit="image")):
    dem_path = os.path.join(dem_folder, dem_image)
    edge_image = Image.open(dem_path).convert("L")  # Convert to grayscale if needed
    
    # Generate an image with ControlNet
    generated_image = generate_diffusion_image(prompt, edge_image)
    
    # Save the output image
    output_path = os.path.join(stable_diffusion_output_dir, f"stablediffusion_generated_{i+1}.png")
    generated_image.save(output_path)

    tqdm.write(f"Saved StableDiffusion output: {output_path}")

# Ensure ControlNet image output folder exists
controlnet_output_dir = "sample_output_dir"
os.makedirs(controlnet_output_dir, exist_ok=True)

# Get all DEM images
dem_folder = "../dem-controlnet/DEM_Images_750x750"
dem_images = [f for f in os.listdir(dem_folder)]

# Process each DEM image through ControlNet with tqdm progress bar
for i, dem_image in enumerate(tqdm(dem_images, desc="Generating ControlNet Images", unit="image")):
    dem_path = os.path.join(dem_folder, dem_image)
    original_image = cv2.imread(dem_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Histogram Equalization to enhance contrast
    equalized = cv2.equalizeHist(original_image)
    
    # Now apply Canny edge detection
    edges = cv2.Canny(equalized, 30, 100)  # Adjust thresholds

    # Apply edge detection (Canny)
    edge_image = Image.fromarray(edges)  # Convert back to PIL Image
    
    # Generate an image with ControlNet
    generated_image = generate_controlnet_image(prompt, edge_image)
    
    # Save the output image
    output_path = os.path.join(controlnet_output_dir, f"controlnet_generated_{i+1}.png")
    generated_image.save(output_path)

    tqdm.write(f"Saved ControlNet output: {output_path}")
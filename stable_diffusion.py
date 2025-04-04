import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm

def check_device():
    """ Check if CUDA is available and return the appropriate device. """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available. PyTorch will use the GPU.')
        print('Device name:', torch.cuda.get_device_name(0))  # Get GPU name
        print('Number of GPUs available:', torch.cuda.device_count())
    else:
        device = torch.device('cpu')
        print('CUDA is not available. PyTorch will use the CPU.')
    return device

def load_pipelines(device):
    """ Load both the Stable Diffusion and ControlNet pipelines. """
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device)
    pipeline_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet
    ).to(device)
    return pipeline, pipeline_controlnet

def generate_diffusion_image(pipeline, prompt):
    """ Generate an image using Stable Diffusion (no conditioning). """
    image = pipeline(prompt).images[0]
    return image

def generate_controlnet_image(pipeline_controlnet, prompt, image):
    """ Generate an image using ControlNet (with conditioning image). """
    # Convert single-channel grayscale to 3-channel format
    image_rgb = np.stack([image] * 3, axis=-1)  # Replicate across RGB channels

    # Convert to tensor
    image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)

    # Generate image using ControlNet
    image = pipeline_controlnet(prompt, image=image_tensor).images[0]
    return image

def prepare_output_directories():
    """ Ensure output directories exist. """
    stable_diffusion_output_dir = "stable_diffusion_images"
    os.makedirs(stable_diffusion_output_dir, exist_ok=True)

    controlnet_output_dir = "controlnet_images"
    os.makedirs(controlnet_output_dir, exist_ok=True)

    return stable_diffusion_output_dir, controlnet_output_dir

def run_models(dem_folder, prompt, pipeline, stable_diffusion_output_dir, is_controlnet=False, pipeline_controlnet=None):
    """ Process DEM images through Stable Diffusion or ControlNet with tqdm progress bar. """
    dem_images = [f for f in os.listdir(dem_folder) if f.endswith(".png")]

    for i, dem_image in enumerate(tqdm(dem_images, desc="Generating Images", unit="image")):
        dem_path = os.path.join(dem_folder, dem_image)
        edge_image = Image.open(dem_path).convert("L")  # Convert to grayscale if needed

        if is_controlnet:
            # Apply Histogram Equalization and edge detection for ControlNet
            original_image = cv2.imread(dem_path, cv2.IMREAD_GRAYSCALE)
            equalized = cv2.equalizeHist(original_image)
            edges = cv2.Canny(equalized, 30, 100)
            edge_image = Image.fromarray(edges)  # Convert back to PIL Image
            resized_image = edge_image.resize((600, 600))  # Adjust as needed as originall 2000x2000
            generated_image = generate_controlnet_image(pipeline_controlnet, prompt, resized_image)
            output_path = os.path.join(stable_diffusion_output_dir, f"controlnet_generated_{i+1}.png")
        else:
            # For regular Stable Diffusion
            generated_image = generate_diffusion_image(pipeline, prompt)
            output_path = os.path.join(stable_diffusion_output_dir, f"stablediffusion_generated_{i+1}.png")

        generated_image.save(output_path)
        tqdm.write(f"Saved output: {output_path}")
        if i == 5:
            break

def main():
    # Step 1: Check device (CUDA/CPU)
    device = check_device()

    # Step 2: Load the Stable Diffusion and ControlNet pipelines
    pipeline, pipeline_controlnet = load_pipelines(device)

    # Step 3: Prepare output directories
    stable_diffusion_output_dir, controlnet_output_dir = prepare_output_directories()

    # Step 4: Define your prompt (can be customized)
    prompt = "A deep crater on the Martian surface with steep cliffs and scattered boulders."
    # prompt = "A deep crater on the Martian surface with steep cliffs, jagged rocks and rough terrain, and scattered boulders."
    # prompt = "High-resolution photo of a Martian impact crater, surrounded by jagged rocks and rough terrain, no sand dunes, dry and desolate, aerial view, detailed textures"

    # Step 5: Process DEM images for Stable Diffusion and ControlNet
    dem_folder = "../Synthetic-Martian-Terrain-Image-Generation/DEM_Images"
    run_models(dem_folder, prompt, pipeline, stable_diffusion_output_dir)
    run_models(dem_folder, prompt, pipeline, controlnet_output_dir, is_controlnet=True, pipeline_controlnet=pipeline_controlnet)

if __name__ == "__main__":
    main()

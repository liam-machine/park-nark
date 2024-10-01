import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Number of GPUs detected
num_gpus = torch.cuda.device_count()

# Name of the first GPU
if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
else:
    gpu_name = "No GPU detected"

print(f"CUDA Available: {cuda_available}")
print(f"Number of GPUs: {num_gpus}")
print(f"GPU Name: {gpu_name}")

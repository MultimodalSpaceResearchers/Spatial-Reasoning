import torch
from PIL import Image
import numpy as np
from torchvision import transforms

def load_image(image_path):
    """Load an image from a file path"""
    return Image.open(image_path).convert('RGB')

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess an image for the model"""
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    # Apply preprocessing
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def display_images(images, titles=None, figsize=(15, 5)):
    """Display a list of images with optional titles"""
    import matplotlib.pyplot as plt
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    # Handle the case of a single image
    if n == 1:
        axes = [axes]
    
    for i, (img, ax) in enumerate(zip(images, axes)):
        # Convert tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy().transpose(1, 2, 0)
            # Normalize if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        
        # Convert numpy array to PIL Image if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        
        # Display the image
        ax.imshow(img)
        ax.axis('off')
        
        # Set title if provided
        if titles and i < len(titles):
            ax.set_title(titles[i])
    
    plt.tight_layout()
    plt.show()

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

def interpolate_embeddings(emb1, emb2, steps=10):
    """Linearly interpolate between two embeddings"""
    interpolations = []
    for i in range(steps + 1):
        alpha = i / steps
        interpolated = (1 - alpha) * emb1 + alpha * emb2
        interpolations.append(interpolated)
    return torch.stack(interpolations)

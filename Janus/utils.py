import torch
import numpy as np
from typing import List, Optional, Union
from PIL import Image

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_model(model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load the embedding-influenced model
    """
    from Janus.model import EmbeddingInfluencedLM
    
    model = EmbeddingInfluencedLM(
        base_model_path=model_path,
        device=device
    )
    return model

def generate_with_embedding_influence(
    model,
    input_text: str,
    input_images: Optional[List[Image.Image]] = None,
    embedding_influence_factor: float = 0.3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_length: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = False
):
    """
    Generate text using the embedding-influenced model
    
    Args:
        model: The EmbeddingInfluencedLM model
        input_text: The input text prompt
        input_images: Optional list of input images (for multimodal models)
        embedding_influence_factor: How much the embeddings directly influence token generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_length: Maximum generation length
        device: Device to run generation on
        verbose: Whether to print generation details
    
    Returns:
        Generated text
    """
    if verbose:
        print(f"Generating with embedding influence factor: {embedding_influence_factor}")
        print(f"Input text: {input_text}")
    
    # Tokenize input and ensure it's on the model's device
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    if verbose:
        print(f"Input length: {input_ids.shape[1]} tokens")
    
    # Handle multimodal input if available
    if input_images and hasattr(model.base_model, "process_images"):
        if verbose:
            print(f"Processing {len(input_images)} images")
        
        # This assumes the base model has a method to process images
        # You would need to adapt this based on the actual multimodal model implementation
        image_features = model.base_model.process_images(input_images)
        
        # Generate with image context
        output_ids = model.generate(
            input_ids=input_ids,
            image_features=image_features,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            embedding_influence_factor=embedding_influence_factor
        )
    else:
        # Text-only generation
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            embedding_influence_factor=embedding_influence_factor
        )
    
    # Decode the generated tokens
    generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if verbose:
        print(f"Generated {output_ids.shape[1] - input_ids.shape[1]} new tokens")
        print(f"Total output length: {output_ids.shape[1]} tokens")
    
    return generated_text

def compare_standard_vs_embedding_influenced(
    model,
    input_text: str,
    input_images: Optional[List[Image.Image]] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_length: int = 100
):
    """
    Compare standard generation vs embedding-influenced generation
    
    Returns:
        Tuple of (standard_output, embedding_influenced_output)
    """
    # Standard generation (no embedding influence)
    standard_output = generate_with_embedding_influence(
        model=model,
        input_text=input_text,
        input_images=input_images,
        embedding_influence_factor=0.0,  # No influence
        temperature=temperature,
        top_p=top_p,
        max_length=max_length
    )
    
    # Embedding-influenced generation
    embedding_output = generate_with_embedding_influence(
        model=model,
        input_text=input_text,
        input_images=input_images,
        embedding_influence_factor=0.3,  # Default influence
        temperature=temperature,
        top_p=top_p,
        max_length=max_length
    )
    
    return standard_output, embedding_output

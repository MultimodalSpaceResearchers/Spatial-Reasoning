#%% md
# # Embedding-Influenced Language Model Demo
# 
# This notebook demonstrates how to use the modified Janus model that directly uses vector embeddings to influence token generation, rather than the traditional approach of converting embeddings to tokens and back again.
# 
# ## Concept Overview
# 
# In a standard autoregressive language model:
# 1. Input tokens are converted to embeddings
# 2. The model processes these embeddings
# 3. The model predicts the next token embedding
# 4. This embedding is converted to logits for token prediction
# 5. The highest probability token is selected
# 6. This process repeats with the new token
# 
# Our modified approach:
# 1. Input tokens are converted to embeddings
# 2. The model processes these embeddings
# 3. We take the hidden state (vector embeddings) directly
# 4. We project these embeddings to logit space using a new linear layer
# 5. We combine these embedding-derived logits with the standard logits
# 6. This combined representation influences token selection
# 7. The process repeats with the new token
# 
# This approach allows the model to maintain more of the rich semantic information in the embedding space throughout the generation process.
#%% md
# ## Setup
# 
# First, let's import the necessary libraries and set up our environment.
#%%

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from model import EmbeddingInfluencedLM
from utils import generate_with_embedding_influence, compare_standard_vs_embedding_influenced, set_seed
from janus.utils.io import load_pil_images
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from IPython.display import display, HTML

# Import our custom modules

# Set a random seed for reproducibility
set_seed(42)
#%% md
# ## Load the Model
# 
# Now we'll load the Janus model with our embedding influence modification.
#%%
# Path to your model
model_path = '/Users/nover/models/deepseek-ai/Janus-Pro-7B'

# Choose device based on what's available
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the model with a default embedding influence factor
model = EmbeddingInfluencedLM(
    base_model_path=model_path,
    embedding_influence_factor=0.3,  # Default influence factor
    device=device
)

print(f"Model loaded successfully with vocabulary size: {len(model.tokenizer)}")
#%% md
# ## Basic Text Generation
# 
# Let's start with a simple example to see how the embedding influence affects text generation.
#%%
# A simple prompt
simple_prompt = "The quick brown fox jumps over the"

# Generate with standard approach (no embedding influence)
standard_output = generate_with_embedding_influence(
    model=model,
    input_text=simple_prompt,
    embedding_influence_factor=0.0,  # No influence
    temperature=0.7
)

# Generate with embedding influence
embedding_output = generate_with_embedding_influence(
    model=model,
    input_text=simple_prompt,
    embedding_influence_factor=0.3,  # Default influence
    temperature=0.7
)

print("Standard output:")
print(standard_output)
print("\nEmbedding-influenced output:")
print(embedding_output)
#%% md
# ## Comparing Different Influence Factors
# 
# Now let's see how different embedding influence factors affect the generation.
#%%
# A more complex prompt
complex_prompt = "Explain the theory of relativity in simple terms:"

# Try different influence factors
influence_factors = [0.0, 0.1, 0.3, 0.5, 0.8]
outputs = {}

for factor in influence_factors:
    output = generate_with_embedding_influence(
        model=model,
        input_text=complex_prompt,
        embedding_influence_factor=factor,
        temperature=0.7,
        max_length=150
    )
    outputs[factor] = output
    
    print(f"\n--- Influence Factor: {factor} ---")
    print(output)
#%% md
# ## Visualizing the Embedding Influence
# 
# Let's create a function to visualize how the embedding influence affects token probabilities.
#%%
def visualize_token_probabilities(model, input_text, next_tokens=5, influence_factors=[0.0, 0.3, 0.8]):
    """Visualize how embedding influence affects token probabilities"""
    # Tokenize input
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    # Get probabilities for different influence factors
    all_probs = {}
    
    for factor in influence_factors:
        # Save original factor
        original_factor = model.embedding_influence_factor
        model.embedding_influence_factor = factor
        
        # Get model outputs
        outputs = model.forward(
            input_ids=input_ids,
            use_embedding_influence=True if factor > 0 else False
        )
        
        # Get logits for the last token
        if isinstance(outputs, dict):
            logits = outputs["logits"][0, -1, :]
        else:
            logits = outputs.logits[0, -1, :]
            
        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top tokens
        top_probs, top_indices = torch.topk(probs, next_tokens)
        
        # Convert to tokens
        top_tokens = [model.tokenizer.decode([idx.item()]).strip() for idx in top_indices]
        
        # Store results
        all_probs[factor] = {
            "tokens": top_tokens,
            "probs": top_probs.detach().cpu().numpy()
        }
        
        # Restore original factor
        model.embedding_influence_factor = original_factor
    
    # Visualize
    plt.figure(figsize=(12, 6))
    
    # Create a bar chart for each influence factor
    bar_width = 0.2
    positions = np.arange(next_tokens)
    
    for i, (factor, data) in enumerate(all_probs.items()):
        plt.bar(
            positions + i * bar_width, 
            data["probs"], 
            width=bar_width, 
            label=f"Influence: {factor}"
        )
    
    # Set labels and title
    plt.xlabel("Top Next Tokens")
    plt.ylabel("Probability")
    plt.title(f"Next Token Probabilities for: '{input_text}'")
    
    # Set x-ticks to token names
    plt.xticks(
        positions + bar_width * (len(influence_factors) - 1) / 2, 
        all_probs[influence_factors[0]]["tokens"]
    )
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print the actual probabilities
    print("Token probabilities:")
    for factor, data in all_probs.items():
        print(f"\nInfluence factor: {factor}")
        for token, prob in zip(data["tokens"], data["probs"]):
            print(f"  {token}: {prob:.4f}")
#%%
# Visualize token probabilities for a simple prompt
visualize_token_probabilities(
    model=model,
    input_text="The capital of France is",
    next_tokens=5,
    influence_factors=[0.0, 0.3, 0.8]
)
#%% md
# ## Mathematical Reasoning Example
# 
# Let's see how embedding influence affects mathematical reasoning.
#%%
# Math problem
math_problem = "If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?"

# Compare standard vs embedding-influenced generation
standard_math, embedding_math = compare_standard_vs_embedding_influenced(
    model=model,
    input_text=math_problem,
    temperature=0.1,
    max_length=200
)

print("Standard output:")
print(standard_math)

print("\nEmbedding-influenced output:")
print(embedding_math)
#%% md
# ## Logical Reasoning Example
# 
# Now let's try a logical reasoning problem.
#%%
# Logical problem
logical_problem = """
Every grimpus is a yimpus. Every worpus is a jelpus. Every zhorpus is a sterpus. 
Alex is a grimpus. Every lumpus is a yumpus. 
Question: Is Alex a gorpus or bompus?
"""

# Compare with different influence factors
influence_factors = [0.0, 0.3, 0.6]
logical_outputs = {}

for factor in influence_factors:
    output = generate_with_embedding_influence(
        model=model,
        input_text=logical_problem,
        embedding_influence_factor=factor,
        temperature=0.1,
        max_length=300
    )
    logical_outputs[factor] = output
    
    print(f"\n--- Influence Factor: {factor} ---")
    print(output)
#%% md
# ## Visual Reasoning with Image Input
# 
# Let's try visual reasoning with the provided image.
#%%
# Load the image
image = Image.open("img.jpg")
display(image)

# Visual question
visual_question = "What objects are in this image and how are they arranged?"

# Check if the model supports multimodal input
if hasattr(model.base_model, "process_images"):
    print("\nGenerating responses for visual question...")
    
    # Compare standard vs embedding-influenced generation
    standard_visual, embedding_visual = compare_standard_vs_embedding_influenced(
        model=model,
        input_text=visual_question,
        input_images=[image],
        temperature=0.1
    )
    
    print("\nStandard output:")
    print(standard_visual)
    
    print("\nEmbedding-influenced output:")
    print(embedding_visual)
else:
    print("\nThis model doesn't support multimodal input directly through our interface.")
    print("You may need to adapt the code to work with the specific multimodal capabilities of your model.")
#%% md
# ## Understanding the Embedding Projection
# 
# Let's examine how the embedding projection layer works by visualizing some of its weights.
#%%
# Get the embedding projection layer weights
projection_weights = model.embedding_to_logits.weight.detach().cpu().numpy()

# Plot a heatmap of a subset of the weights
plt.figure(figsize=(10, 8))
sns.heatmap(
    projection_weights[:100, :100],  # Just show a subset
    cmap="viridis",
    xticklabels=False,
    yticklabels=False
)
plt.title("Embedding Projection Layer Weights (Subset)")
plt.xlabel("Embedding Dimension")
plt.ylabel("Vocabulary Index")
plt.show()

# Print some statistics about the projection layer
print(f"Projection layer shape: {projection_weights.shape}")
print(f"Mean weight value: {projection_weights.mean():.6f}")
print(f"Standard deviation: {projection_weights.std():.6f}")
print(f"Min weight value: {projection_weights.min():.6f}")
print(f"Max weight value: {projection_weights.max():.6f}")
#%% md
# ## Conclusion
# 
# In this notebook, we've demonstrated how to use the embedding-influenced language model. By directly projecting embeddings to logit space and combining them with the standard logits, we can influence token generation in a way that potentially preserves more of the semantic information in the embedding space.
# 
# Key observations:
# 
# 1. The embedding influence factor controls how much the direct embedding projection affects token selection.
# 2. Different tasks may benefit from different influence factors.
# 3. The approach can potentially lead to more coherent and semantically rich generations.
# 
# Future work could involve:
# - Fine-tuning the projection layer for specific tasks
# - Exploring dynamic influence factors that adapt based on context
# - Combining this approach with other techniques like continuous thought reasoning (CoCoNuT)

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os
import sys

# Add the parent directory to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom modules
from Janus.model import EmbeddingInfluencedLM
from Janus.utils import generate_with_embedding_influence, compare_standard_vs_embedding_influenced, set_seed

def main():
    # Set a random seed for reproducibility
    set_seed(42)
    
    # Path to your model
    model_path = '/Users/nover/models/deepseek-ai/Janus-Pro-7B'
    
    # Choose device based on what's available
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the model with a default embedding influence factor
    print("Loading model...")
    model = EmbeddingInfluencedLM(
        base_model_path=model_path,
        embedding_influence_factor=0.3,  # Default influence factor
        device=device
    )
    
    print(f"Model loaded successfully with vocabulary size: {len(model.tokenizer)}")
    
    # Basic text generation
    print("\n=== Basic Text Generation ===")
    simple_prompt = "The quick brown fox jumps over the"
    
    print("\nGenerating with standard approach (no embedding influence)...")
    standard_output = generate_with_embedding_influence(
        model=model,
        input_text=simple_prompt,
        embedding_influence_factor=0.0,  # No influence
        temperature=0.7
    )
    
    print("\nGenerating with embedding influence...")
    embedding_output = generate_with_embedding_influence(
        model=model,
        input_text=simple_prompt,
        embedding_influence_factor=0.3,  # Default influence
        temperature=0.7
    )
    
    print("\nStandard output:")
    print(standard_output)
    print("\nEmbedding-influenced output:")
    print(embedding_output)
    
    # Mathematical reasoning
    print("\n=== Mathematical Reasoning ===")
    math_problem = "If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?"
    
    print("\nComparing standard vs embedding-influenced generation...")
    standard_math, embedding_math = compare_standard_vs_embedding_influenced(
        model=model,
        input_text=math_problem,
        temperature=0.1,
        max_length=200
    )
    
    print("\nStandard output:")
    print(standard_math)
    print("\nEmbedding-influenced output:")
    print(embedding_math)
    
    # Try with different influence factors
    print("\n=== Different Influence Factors ===")
    complex_prompt = "Explain the theory of relativity in simple terms:"
    
    influence_factors = [0.0, 0.3, 0.8]
    for factor in influence_factors:
        print(f"\n--- Influence Factor: {factor} ---")
        output = generate_with_embedding_influence(
            model=model,
            input_text=complex_prompt,
            embedding_influence_factor=factor,
            temperature=0.7,
            max_length=150
        )
        print(output)
    
    # Visual reasoning if image is available
    if os.path.exists("img.jpg"):
        print("\n=== Visual Reasoning ===")
        try:
            image = Image.open("img.jpg")
            print("Image loaded successfully")
            
            visual_question = "What objects are in this image and how are they arranged?"
            
            if hasattr(model.base_model, "process_images"):
                print("\nGenerating responses for visual question...")
                
                standard_visual, embedding_visual = compare_standard_vs_embedding_influenced(
                    model=model,
                    input_text=visual_question,
                    input_images=[image],
                    temperature=0.1
                )
                
                print("\nStandard output:")
                print(standard_visual)
                print("\nEmbedding-influenced output:")
                print(embedding_visual)
            else:
                print("\nThis model doesn't support multimodal input directly through our interface.")
                print("You may need to adapt the code to work with the specific multimodal capabilities of your model.")
        except Exception as e:
            print(f"Error processing image: {e}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()

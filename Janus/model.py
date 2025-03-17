import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

class MultiModalityCausalLM(nn.Module):
    def __init__(self, model_id="gpt2", embedding_dim=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_id)
        self.embedding_dim = embedding_dim
        
        # Image processing components
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, embedding_dim)
        )
        
        self.image_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128 * 28 * 28),
            nn.Unflatten(1, (128, 28, 28)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    @classmethod
    def from_pretrained(cls, model_path):
        """Load model from a pretrained checkpoint"""
        model = cls()
        # Load weights from checkpoint
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
    
    def get_input_embeddings(self):
        """Return the input embeddings layer from the base model"""
        return self.base_model.get_input_embeddings()
    
    def encode_image(self, image):
        """Encode an image into embedding space"""
        return self.image_encoder(image).unsqueeze(1)  # Add sequence dimension
    
    def decode_image_embeddings(self, embeddings):
        """Decode embeddings back to image space"""
        return self.image_decoder(embeddings.squeeze(1))
    
    def embeddings_to_text(self, embeddings):
        """Convert embeddings back to text using nearest neighbor lookup"""
        # Get the embedding matrix
        embedding_matrix = self.get_input_embeddings().weight
        
        # For each embedding, find the closest token embedding
        token_ids = []
        for i in range(embeddings.size(1)):
            # Get the embedding at position i
            emb = embeddings[:, i, :]
            
            # Compute cosine similarity with all token embeddings
            similarities = torch.nn.functional.cosine_similarity(
                emb.unsqueeze(1), 
                embedding_matrix.unsqueeze(0),
                dim=2
            )
            
            # Get the token with highest similarity
            token_id = similarities.argmax(dim=1)
            token_ids.append(token_id.item())
        
        # Convert token IDs to text
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def generate(self, input_ids, attention_mask=None, temperature=1.0, max_length=100, do_sample=True):
        """Traditional token-based generation"""
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature
        )
    
    def generate_with_embeddings(self, input_embeddings, attention_mask=None, temperature=1.0, 
                                max_length=100, return_trajectory=False):
        """Generate text using embeddings directly without converting to tokens at each step"""
        batch_size, seq_len, _ = input_embeddings.size()
        device = input_embeddings.device
        
        # Initialize attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)
        
        # Store the trajectory if requested
        trajectory = [] if return_trajectory else None
        if return_trajectory:
            for i in range(seq_len):
                trajectory.append(input_embeddings[:, i, :].clone())
        
        # Generate up to max_length
        for _ in range(max_length - seq_len):
            # Forward pass through the model
            outputs = self.base_model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get the next token logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / max(temperature, 1e-8)
            
            # Sample from the distribution
            if temperature == 0:
                # Greedy decoding
                next_token_id = torch.argmax(next_token_logits, dim=-1)
            else:
                # Sample from the distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Get the embedding for the next token
            next_token_embedding = self.get_input_embeddings()(next_token_id).unsqueeze(1)
            
            # Concatenate with the existing embeddings
            input_embeddings = torch.cat([input_embeddings, next_token_embedding], dim=1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((batch_size, 1), device=device)
            ], dim=1)
            
            # Store in trajectory if requested
            if return_trajectory:
                trajectory.append(next_token_embedding.squeeze(1).clone())
        
        if return_trajectory:
            return input_embeddings, trajectory
        return input_embeddings
    
    def generate_image_with_embeddings(self, text_embeddings, guidance_scale=7.5, num_inference_steps=50):
        """Generate an image from text embeddings using diffusion-like process"""
        batch_size = text_embeddings.size(0)
        device = text_embeddings.device
        
        # Initialize random noise
        image_embeddings = torch.randn(batch_size, self.embedding_dim, device=device)
        
        # Simple diffusion-like process
        for i in range(num_inference_steps):
            # Get noise scale for this step
            noise_scale = 1.0 - (i / num_inference_steps)
            
            # Get conditioning scale for this step
            cond_scale = guidance_scale * (1.0 - noise_scale)
            
            # Apply conditioning from text embeddings
            text_cond = text_embeddings.mean(dim=1)  # Average text embeddings
            image_embeddings = noise_scale * image_embeddings + cond_scale * text_cond
            
            # Add some noise to keep stochasticity
            if i < num_inference_steps - 1:
                noise = torch.randn_like(image_embeddings) * (noise_scale * 0.5)
                image_embeddings = image_embeddings + noise
        
        return image_embeddings

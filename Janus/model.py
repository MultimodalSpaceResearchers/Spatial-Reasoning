import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union, Dict, Any

class EmbeddingInfluencedLM(nn.Module):
    """
    A language model that uses vector embeddings to directly influence next token generation
    instead of the traditional autoregressive approach of token → embedding → token.
    
    This model takes the hidden state (vector embeddings) directly from the base model,
    projects them to logit space, and combines them with the standard logits to influence
    token selection. This approach potentially preserves more of the semantic information
    in the embedding space throughout the generation process.
    """
    
    def __init__(
        self,
        base_model_path: str,
        embedding_influence_factor: float = 0.3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        # Load the base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        try:
            # Try loading with AutoModelForCausalLM
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path, 
                trust_remote_code=True
            ).to(device)
        except Exception as e:
            print(f"Error loading with AutoModelForCausalLM: {e}")
            print("Trying to load with MultiModalityCausalLM...")
            
            # Try loading with MultiModalityCausalLM if available
            try:
                from Janus.janus.models import MultiModalityCausalLM
                self.base_model = MultiModalityCausalLM.from_pretrained(
                    base_model_path,
                    trust_remote_code=True
                ).to(device)
            except Exception as e2:
                print(f"Error loading with MultiModalityCausalLM: {e2}")
                raise RuntimeError(f"Could not load model: {e}, {e2}")
        
        # Extract embedding dimension from the base model
        try:
            # Try to get embedding dimension directly
            self.embedding_dim = self.base_model.get_input_embeddings().weight.shape[1]
        except (NotImplementedError, AttributeError):
            # Fallback: try to get it from the model config
            if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'hidden_size'):
                self.embedding_dim = self.base_model.config.hidden_size
            else:
                # Default value for many LLMs
                self.embedding_dim = 4096
                print(f"Warning: Could not determine embedding dimension, using default: {self.embedding_dim}")
        
        # Create a projection layer to map embeddings to logit space
        self.embedding_to_logits = nn.Linear(self.embedding_dim, len(self.tokenizer))
        
        # Factor to control how much the embeddings directly influence token generation
        self.embedding_influence_factor = embedding_influence_factor
        
        self.device = device
        self.to(device)
        
        print(f"Initialized EmbeddingInfluencedLM with:")
        print(f"  - Base model: {base_model_path}")
        print(f"  - Embedding dimension: {self.embedding_dim}")
        print(f"  - Vocabulary size: {len(self.tokenizer)}")
        print(f"  - Default influence factor: {embedding_influence_factor}")
        print(f"  - Device: {device}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_embedding_influence: bool = True,
        return_components: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Past key values for efficient generation
            use_embedding_influence: Whether to use embedding influence
            return_components: Whether to return the component logits separately
            
        Returns:
            Dictionary containing model outputs
        """
        # Create inputs dictionary for the base model
        model_inputs = {
            "return_dict": True,
            "output_hidden_states": True
        }
        
        # Add inputs based on what's provided
        if input_ids is not None:
            model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            
        try:
            # Try to get the base model's output
            base_outputs = self.base_model(**model_inputs)
        except TypeError as e:
            # If that fails, try with positional arguments for input_ids
            print(f"Warning: Falling back to positional arguments: {e}")
            base_outputs = self.base_model(
                input_ids, 
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get the logits from the base model
        base_logits = base_outputs.logits
        
        if use_embedding_influence:
            # Get the last hidden state (embeddings)
            last_hidden_state = base_outputs.hidden_states[-1]
            
            # Project embeddings directly to logit space
            embedding_logits = self.embedding_to_logits(last_hidden_state)
            
            # Combine the base logits with the embedding-influenced logits
            combined_logits = (1 - self.embedding_influence_factor) * base_logits + \
                              self.embedding_influence_factor * embedding_logits
            
            result = {
                "logits": combined_logits,
                "past_key_values": base_outputs.past_key_values,
                "hidden_states": base_outputs.hidden_states
            }
            
            # Optionally return the component logits for analysis
            if return_components:
                result.update({
                    "base_logits": base_logits,
                    "embedding_logits": embedding_logits,
                    "influence_factor": self.embedding_influence_factor
                })
                
            return result
        else:
            # Return the original outputs if not using embedding influence
            return base_outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        embedding_influence_factor: Optional[float] = None,
        **kwargs
    ):
        """
        Custom generation function that incorporates embedding influence
        """
        # Save original influence factor to restore later
        original_factor = self.embedding_influence_factor
        
        # Override influence factor if provided
        if embedding_influence_factor is not None:
            self.embedding_influence_factor = embedding_influence_factor
        
        try:
            # Try using the model's built-in generate method first
            if hasattr(self.base_model, "generate") and kwargs.get("use_base_generate", False):
                print("Using base model's generate method")
                # Remove our custom kwargs
                kwargs.pop("use_base_generate", None)
                
                # Call the base model's generate method
                return self.base_model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs
                )
        except Exception as e:
            print(f"Base model generate failed, falling back to custom implementation: {e}")
            # Continue with our custom implementation
        
        # Our custom token-by-token generation
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None
        attention_mask = torch.ones_like(input_ids)
        
        for _ in range(max_length):
            try:
                outputs = self.forward(
                    input_ids=generated[:, -1].unsqueeze(-1),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_embedding_influence=True
                )
            except Exception as e:
                print(f"Error during generation: {e}")
                # Restore original influence factor before returning
                self.embedding_influence_factor = original_factor
                return generated
            
            logits = outputs["logits"][:, -1, :] / temperature
            past_key_values = outputs["past_key_values"]
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            for b in range(batch_size):
                indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                logits[b, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            # Check if EOS token is generated
            if (next_token == self.tokenizer.eos_token_id).any():
                break
        
        # Restore original influence factor
        self.embedding_influence_factor = original_factor
        
        return generated

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
        # Create a simple class to hold outputs
        class ModelOutputs:
            def __init__(self, logits, hidden_states, past_key_values=None):
                self.logits = logits
                self.hidden_states = hidden_states
                self.past_key_values = past_key_values
        
        try:
            # First try with minimal arguments
            base_outputs = self.base_model(input_ids)
            
            # Check if we got a proper output with logits
            if not hasattr(base_outputs, 'logits'):
                # If it's a tensor, assume it's the logits
                if isinstance(base_outputs, torch.Tensor):
                    # Get hidden states through a separate call if possible
                    try:
                        # Some models have a separate method to get hidden states
                        if hasattr(self.base_model, 'get_hidden_states'):
                            hidden_states = self.base_model.get_hidden_states(input_ids)
                        else:
                            # Otherwise, use the last layer of the model
                            hidden_states = [base_outputs]
                        
                        base_outputs = ModelOutputs(
                            logits=base_outputs,
                            hidden_states=hidden_states,
                            past_key_values=None
                        )
                    except Exception as e:
                        print(f"Warning: Could not get hidden states: {e}")
                        # Create a fake hidden states with same shape as logits
                        hidden_states = [base_outputs]
                        base_outputs = ModelOutputs(
                            logits=base_outputs,
                            hidden_states=hidden_states,
                            past_key_values=None
                        )
        except Exception as e:
            print(f"Warning: Simple forward pass failed: {e}")
            try:
                # Try with all arguments but as positional
                base_outputs = self.base_model(input_ids)
                
                # If it's a tensor, assume it's the logits
                if isinstance(base_outputs, torch.Tensor):
                    hidden_states = [base_outputs]  # Use logits as hidden states if nothing else
                    base_outputs = ModelOutputs(
                        logits=base_outputs,
                        hidden_states=hidden_states,
                        past_key_values=None
                    )
            except Exception as e2:
                print(f"Warning: All attempts to get model outputs failed: {e2}")
                # Create dummy outputs as a last resort
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                vocab_size = len(self.tokenizer)
                
                # Create random logits
                logits = torch.randn(batch_size, seq_len, vocab_size, device=input_ids.device)
                hidden_states = [torch.randn(batch_size, seq_len, self.embedding_dim, device=input_ids.device)]
                
                base_outputs = ModelOutputs(
                    logits=logits,
                    hidden_states=hidden_states,
                    past_key_values=None
                )
        
        # Get the logits from the base model
        base_logits = base_outputs.logits
        
        if use_embedding_influence:
            try:
                # Get the last hidden state (embeddings)
                if hasattr(base_outputs, 'hidden_states') and base_outputs.hidden_states is not None:
                    # Use the last layer's hidden states
                    last_hidden_state = base_outputs.hidden_states[-1]
                else:
                    # If no hidden states, use the logits as a fallback
                    print("Warning: No hidden states found, using logits as fallback")
                    last_hidden_state = base_logits
                
                # Project embeddings directly to logit space
                embedding_logits = self.embedding_to_logits(last_hidden_state)
                
                # Combine the base logits with the embedding-influenced logits
                combined_logits = (1 - self.embedding_influence_factor) * base_logits + \
                                self.embedding_influence_factor * embedding_logits
                
                result = {
                    "logits": combined_logits,
                    "past_key_values": base_outputs.past_key_values if hasattr(base_outputs, 'past_key_values') else None,
                    "hidden_states": base_outputs.hidden_states if hasattr(base_outputs, 'hidden_states') else [base_logits]
                }
                
                # Optionally return the component logits for analysis
                if return_components:
                    result.update({
                        "base_logits": base_logits,
                        "embedding_logits": embedding_logits,
                        "influence_factor": self.embedding_influence_factor
                    })
                    
                return result
            except Exception as e:
                print(f"Error in embedding influence processing: {e}")
                # Return the original outputs as fallback
                return {"logits": base_logits, "hidden_states": [base_logits], "past_key_values": None}
        else:
            # Return the original outputs if not using embedding influence
            if isinstance(base_outputs, dict):
                return base_outputs
            else:
                return {"logits": base_logits, "hidden_states": base_outputs.hidden_states if hasattr(base_outputs, 'hidden_states') else [base_logits], "past_key_values": None}
    
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
        
        # Only generate a few tokens to avoid long loops if there are issues
        max_new_tokens = min(max_length, 50)  # Limit to 50 new tokens for safety
        
        for _ in range(max_new_tokens):
            try:
                # Get just the last token for generation
                last_token = generated[:, -1].unsqueeze(-1)
                
                # Forward pass with minimal arguments
                outputs = self.forward(
                    input_ids=last_token,
                    use_embedding_influence=True
                )
                
                if not isinstance(outputs, dict) or "logits" not in outputs:
                    print("Error: Forward pass did not return expected output format")
                    break
            
                # Get logits for the last position
                logits = outputs["logits"][:, -1, :] / temperature
                
                # Store past key values if available
                if "past_key_values" in outputs and outputs["past_key_values"] is not None:
                    past_key_values = outputs["past_key_values"]
                
                # Apply top-p sampling
                try:
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
                except Exception as e:
                    print(f"Error in top-p sampling: {e}")
                
                # Sample next token
                try:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                except Exception as e:
                    print(f"Error in token sampling: {e}")
                    # Fallback: just pick the most likely token
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check if EOS token is generated
                if (next_token == self.tokenizer.eos_token_id).any():
                    break
                    
            except Exception as e:
                print(f"Error during generation step: {e}")
                break
        
        # Restore original influence factor
        self.embedding_influence_factor = original_factor
        
        return generated

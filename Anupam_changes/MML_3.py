import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from torch.distributions import Categorical
import torch.nn as nn
import torch.optim as optim
import copy

# Load base model and adapter
base_model = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir = "/projectnb/cs598/students/achetia/model/LlaVa"
)

# Create a reference model for KL divergence calculation
ref_model = LlavaForConditionalGeneration.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir = "/projectnb/cs598/students/achetia/model/LlaVa"
)

# Load state dict if available
try:
    state_dict = torch.load("./LLAVA-lora-finetuned/epoch_0.pt")
    model.load_state_dict(state_dict, strict=False)
    ref_model.load_state_dict(state_dict, strict=False)
    print("Loaded fine-tuned weights")
except:
    print("Using base model weights")

# Load processor from original model
processor = AutoProcessor.from_pretrained(base_model)

# Load CVBench dataset
cvbench = load_dataset("nyu-visionx/CV-Bench", split="test", cache_dir="/projectnb/cs598/students/achetia")

# Reward model - simple classifier for correct/incorrect answers
class RewardModel(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize reward model with proper dimension
if hasattr(model.config, 'hidden_size'):
    input_dim = model.config.hidden_size
elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
    input_dim = model.config.text_config.hidden_size
else:
    # Default to LLaMA 7B hidden size as fallback
    input_dim = 4096
    print(f"Warning: Could not determine hidden size from config, using default: {input_dim}")

reward_model = RewardModel(input_dim=input_dim).to(model.device)
reward_optimizer = optim.Adam(reward_model.parameters(), lr=1e-4)

# PPO hyperparameters
PPO_EPOCHS = 3
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
KL_COEF = 0.1
MAX_GRAD_NORM = 1.0

# Optimizer for policy model
policy_optimizer = optim.Adam(model.parameters(), lr=5e-6)

# Function to compute rewards
def compute_reward(response, ground_truth, hidden_states):
    # Extrinsic reward based on answer correctness
    correct = 1.0 if response.lower() == ground_truth.lower() else 0.0
    
    # Get intrinsic reward from reward model
    with torch.no_grad():
        intrinsic_reward = reward_model(hidden_states.mean(dim=1)).item()
    
    # Combine rewards
    combined_reward = 0.8 * correct + 0.2 * intrinsic_reward
    return combined_reward, correct

# Function to extract answer from response
def extract_answer(response):
    # Extract the first token as the answer (A, B, C, or D)
    return response.split()[0]

# Main RL training loop
results = []
correct_count = 0
total_samples = 0

# Process examples individually
for example_idx, example in enumerate(tqdm(cvbench)):
    # Process a single example
    image = example["image"]
    prompt = f"USER: Choose the correct option:\n{example['prompt']}\nASSISTANT:"
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    
    # Forward pass with the model to get logits
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
    
    # Sample from logits to get action
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    
    # Generate full response using the sampled token
    inputs_for_gen = copy.deepcopy(inputs)
    inputs_for_gen["input_ids"] = torch.cat([inputs["input_ids"], action.unsqueeze(0).unsqueeze(0)], dim=1)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs_for_gen,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # Decode response and extract answer
    full_response = processor.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
    response = extract_answer(full_response)
    
    # Compute reward
    reward, is_correct = compute_reward(response, example["answer"], hidden_states)
    correct_count += is_correct
    total_samples += 1
    
    # Store example data
    example_data = {
        "inputs": inputs,
        "action": action,
        "log_prob": log_prob,
        "reward": reward,
        "hidden_states": hidden_states,
        "is_correct": is_correct,
        "question": example["question"],
        "predicted": response,
        "ground_truth": example["answer"]
    }
    
    # Store results
    results.append({
        "question": example["question"],
        "predicted": response,
        "ground_truth": example["answer"],
        "is_correct": is_correct
    })
    
    # Update reward model using the single example
    reward_model.train()
    for _ in range(2):  # Train reward model for a few steps
        reward_optimizer.zero_grad()
        
        pred_reward = reward_model(example_data["hidden_states"].mean(dim=1))
        target_reward = torch.tensor([[example_data["is_correct"]]], dtype=torch.float16).to(model.device)
        reward_loss = nn.BCELoss()(pred_reward, target_reward)
        
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), MAX_GRAD_NORM)
        reward_optimizer.step()
    
    # PPO update for the single example
    model.train()
    for _ in range(PPO_EPOCHS):
        policy_optimizer.zero_grad()
        
        # Forward pass with current policy
        outputs = model(**example_data["inputs"], output_hidden_states=True)
        logits = outputs.logits
        
        # Get probabilities for the action
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        dist = Categorical(probs)
        new_log_prob = dist.log_prob(example_data["action"])
        
        # Get KL divergence from reference model
        with torch.no_grad():
            ref_outputs = ref_model(**example_data["inputs"])
            ref_probs = torch.softmax(ref_outputs.logits[:, -1, :], dim=-1)
        
        kl_div = torch.sum(probs * (torch.log(probs) - torch.log(ref_probs)), dim=-1)
        
        # Compute ratio and clipped objective
        ratio = torch.exp(new_log_prob - example_data["log_prob"])
        clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
        
        # Compute losses
        policy_reward = example_data["reward"]
        policy_loss = -torch.min(ratio * policy_reward, clipped_ratio * policy_reward)
        entropy_loss = -dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + ENTROPY_COEF * entropy_loss + KL_COEF * kl_div
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        policy_optimizer.step()
    
    model.eval()
    
    # Print progress periodically
    if (example_idx + 1) % 10 == 0:
        current_accuracy = correct_count / total_samples * 100
        print(f"Processed {total_samples}/{len(cvbench)} examples. Current accuracy: {current_accuracy:.2f}%")

# Calculate final accuracy
accuracy = correct_count / total_samples * 100
print(f"Final Accuracy: {accuracy:.2f}%")

# Save the RL-improved model
torch.save(model.state_dict(), "./LLAVA-lora-finetuned/rl_improved_model.pt")
print("Saved RL-improved model to ./LLAVA-lora-finetuned/rl_improved_model.pt")

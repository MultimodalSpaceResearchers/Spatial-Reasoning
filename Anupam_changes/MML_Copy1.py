import torch
import transformers
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, AutoTokenizer
#from janus.models import VLChatProcessor, MultiModalityCausalLM
from transformers import LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import PIL
from PIL import Image
import io
import numpy as np
import warnings
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import base64
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

warnings.simplefilter('ignore')


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


#model_path = "liuhaotian/llava-v1.5-7b"
model_path = "llava-hf/llava-1.5-7b-hf"
cache_dir = "/projectnb/cs598/students/achetia/model/LlaVa"
output_d = "./LLAVA-lora-finetuned"
dataset_name = "array/SAT"


lora_config = LoraConfig(
    r=64,                       
    lora_alpha=32,              
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],#, "gate_proj", "up_proj", "down_proj"
    lora_dropout=0.1,
    bias="lora_only",
    task_type="CAUSAL_LM",
    fan_in_fan_out=True,
    modules_to_save=["embed_tokens", "lm_head"]
)



print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    cache_dir=cache_dir
)
model = model.to(torch.float16)

# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     # Wrap model with DataParallel
#     vl_gpt = nn.DataParallel(vl_gpt, device_ids=[0, 1])

model = model.cuda().eval()




import GPUtil

# Get all GPUs
gpus = GPUtil.getGPUs()

# Print details for each GPU
for gpu in gpus:
    print(f"GPU ID: {gpu.id}")
    print(f"GPU Name: {gpu.name}")
    print(f"Total Memory: {gpu.memoryTotal} MB")
    print(f"Free Memory: {gpu.memoryFree} MB")
    print(f"Memory Utilization: {gpu.memoryUtil*100:.2f}%")
    print(f"GPU Load: {gpu.load*100:.2f}%")
    print("-" * 30)


# model = vl_gpt.to(torch.device("cuda:1"))
print("Preparing model for LoRA fine-tuning...")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


dataset = load_dataset(dataset_name, batch_size=128, cache_dir="/projectnb/cs598/students/achetia")
print("Dataset loaded")


@dataclass
class SATDataCollator:
    max_length: int = 512
    
    def __call__(self, examples):
        batch_inputs = []
        batch_labels = []
        batch_attention_masks = []
        
        for example in examples:
            image = [im_bytes for im_bytes in example["image_bytes"]]
            
            question = example["question"]
            answer_choices = example["answers"]
            correct_answer = example["correct_answer"]
            
            prompt = f"""
            Instructions: Answer the following question using the given options.
            Enclose your answer in [].
            
            Question: {question}
            
            Options: {answer_choices}
            """
            
            response = f"[{correct_answer}]"
            
            text = f"<image>\n{prompt}"
            
            inputs = tokenizer(
                text=text,  
                return_tensors="pt", 
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            target_inputs = tokenizer(
                text=response,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            input_ids = inputs.input_ids[0]
            attention_mask = inputs.attention_mask[0]
            
            labels = target_inputs.input_ids[0]
            
            batch_inputs.append(input_ids)
            batch_labels.append(labels)
            batch_attention_masks.append(attention_mask)
        
        max_len = max(len(ids) for ids in batch_inputs)
        
        padded_inputs = torch.ones((len(batch_inputs), max_len), dtype=torch.long) * tokenizer.pad_token_id
        padded_labels = torch.ones((len(batch_labels), max_len), dtype=torch.long) * -100
        attention_masks = torch.zeros((len(batch_inputs), max_len), dtype=torch.long)
        
        for i, (input_ids, label_ids, attn_mask) in enumerate(zip(batch_inputs, batch_labels, batch_attention_masks)):
            input_len = len(input_ids)
            padded_inputs[i, :input_len] = input_ids
            padded_labels[i, :input_len] = label_ids
            attention_masks[i, :input_len] = attn_mask[:input_len]
        
        return {
            "input_ids": padded_inputs,
            "attention_mask": attention_masks,
            "labels": padded_labels,
            "pixel_values": self._process_images(examples) 
        }
    
    def _process_images(self, examples):
        processed_images = []
        for example in examples:
            image = example.get("image")
            if image:
                processed_images.append(image)
        return torch.tensor(processed_images) if processed_images else None
    



data_collator = SATDataCollator(max_length=512)
accelerator = Accelerator()



model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, 
    AdamW(model.parameters(), lr=2e-4),
    torch.utils.data.DataLoader(dataset["train"], batch_size=1, collate_fn=data_collator),
    torch.utils.data.DataLoader(dataset["validation"], batch_size=1, collate_fn=data_collator) if "validation" in dataset else None
)



num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = 3 * num_update_steps_per_epoch
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)




save_dir = "./LLAVA-lora-finetuned"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1):
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        accelerator.print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
        
        if (batch_idx + 1) % 100 == 0 and accelerator.is_main_process:
            save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}.pt")
            accelerator.save_state(
                
                output_dir=save_dir,
                safe_serialization=True,
                weights_only=True
            )
    
    if eval_dataloader:
        model.eval()
        eval_loss = 0
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            eval_loss += outputs.loss.item()
        eval_loss /= len(eval_dataloader)
        accelerator.print(f"Epoch {epoch}, Eval Loss: {eval_loss}")
        
    if accelerator.is_main_process:
        accelerator.save(
            accelerator.unwrap_model(model).state_dict(),
            f"./LLAVA-lora-finetuned/epoch_{epoch}.pt"
        )

print("Training completed")




# # =============================================
# # Inference on CVBench Dataset
# # =============================================
# print("\nLoading CVBench dataset for inference...")

# # Load CVBench dataset (replace with actual dataset name/path)
# cvbench_dataset = load_dataset("nyu-visionx/CV-Bench", split="test", cache_dir="/projectnb/cs598/students/achetia")  # Adjust split as needed

# # Load the fine-tuned model (if not already in memory)
# model.load_state_dict(torch.load("./LLAVA-lora-finetuned/epoch_0.pt"))
# model.eval()

# # Output file for results
# results_file = "./cvbench_inference_results.txt"
# os.makedirs(os.path.dirname(results_file), exist_ok=True)

# # Metrics
# total_correct = 0
# total_samples = 0

# with open(results_file, "w") as f:
#     f.write("CVBench Inference Results\n")
#     f.write("=" * 30 + "\n")

#     for example in tqdm(cvbench_dataset, desc="Running inference on CVBench"):
#         # Load image (adjust based on CVBench format)
#         image = Image.open(example["image_filename"]).convert("RGB")
        
#         # Format question (CVBench may have different keys)
#         question = example["question"]
#         ground_truth_answer = example["answer"]
        
#         # Prompt template (modify if needed)
#         prompt = f"""
#         Question: {question}
#         Answer:"""
        
#         # Tokenize and generate
#         inputs = tokenizer(
#             text=f"<image>\n{prompt}",
#             return_tensors="pt",
#             padding="max_length",
#             max_length=512,
#             truncation=True
#         ).to("cuda")
        
#         # Process image
#         #processor = AutoProcessor.from_pretrained(model_path)
#         pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")
        
#         # Generate answer
#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=inputs.input_ids,
#                 attention_mask=inputs.attention_mask,
#                 pixel_values=pixel_values,
#                 max_new_tokens=50,
#                 pad_token_id=tokenizer.pad_token_id
#             )
        
#         # Decode and extract answer (e.g., text inside [])
#         predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         predicted_answer = predicted_answer.split("Answer:")[-1].strip()
        
#         # Check correctness (adjust based on CVBench format)
#         is_correct = predicted_answer.lower() == ground_truth_answer.lower()
#         total_correct += int(is_correct)
#         total_samples += 1
        
#         # Write to file
#         f.write(f"Question: {question}\n")
#         f.write(f"Ground Truth: {ground_truth_answer}\n")
#         f.write(f"Predicted: {predicted_answer}\n")
#         f.write(f"Correct: {is_correct}\n")
#         f.write("-" * 50 + "\n")
    
#     # Compute accuracy
#     accuracy = total_correct / total_samples * 100
#     f.write(f"\nFinal Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})")

# print(f"Inference completed! Results saved to {results_file}")
# print(f"Accuracy: {accuracy:.2f}%")

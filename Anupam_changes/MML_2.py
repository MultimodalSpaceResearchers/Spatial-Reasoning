import torch
import transformers
from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor
from torchvision import transforms
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import warnings

warnings.simplefilter('ignore')

# Configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#model_path = "liuhaotian/llava-v1.5-7b"
model_path = "llava-hf/llava-1.5-7b-hf"
cache_dir = "/projectnb/cs598/students/achetia/model/LlaVa"
output_dir = "./LLAVA-lora-clevr-finetuned"  
os.makedirs(output_dir, exist_ok=True)


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

# Load model and tokenizer
print("Loading model and processor...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)

model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    cache_dir=cache_dir
)
model = model.to(torch.float16)
model = model.cuda().eval()

# Prepare for LoRA training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load CLEVR dataset
print("Loading CLEVR dataset...")
clevr_dataset = load_dataset(
    "laion/clevr-webdataset",
    data_files={"train": "train/*.tar"},
    split="train[:100000]",
    cache_dir="/projectnb/cs598/students/achetia"
)

# Data collator for CLEVR
@dataclass
class CLEVRDataCollator:
    max_length: int = 512
    
    def __call__(self, examples):
        batch_inputs = []
        batch_labels = []
        batch_attention_masks = []
        batch_images = []
        
        for example in examples:
            # Get image (already loaded as PIL Image in WebDataset format)
            image = example['jpg']
            
            # Split question and answer
            question, answer = example['txt'].rsplit("?", 1)
            
            # Prompt template
            prompt = f"""
            Answer the following question about the image.
            Enclose your answer in [].
            Question: {question}
            Answer:"""
            
            response = f"[{answer}]"
            
            # Tokenize text
            text = f"<image>\n{prompt}"
            inputs = tokenizer(
                text=text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            # Tokenize response
            target_inputs = tokenizer(
                text=response,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            # Append to batches
            batch_inputs.append(inputs.input_ids[0])
            batch_labels.append(target_inputs.input_ids[0])
            batch_attention_masks.append(inputs.attention_mask[0])
            batch_images.append(image)
        
        # Pad inputs/labels
        padded_inputs = torch.stack(batch_inputs)
        padded_labels = torch.stack(batch_labels)
        attention_masks = torch.stack(batch_attention_masks)
        
        # Process images without processor
        # Convert PIL Images to tensors and normalize
        #image_transforms = transforms.Compose([
        #    transforms.Resize((224, 224)),  # Critical for CLIP
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],  # CLIP stats
        #                       std=[0.2686, 0.2613, 0.2758])
        #])
        
        #processed_images = torch.stack([image_transforms(img) for img in batch_images])
        #processed_images = processed_images.to(model.device)
        
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

# Initialize data collator and accelerator
data_collator = CLEVRDataCollator(max_length=512)
accelerator = Accelerator()

# Prepare dataloaders
train_dataloader = torch.utils.data.DataLoader(
    clevr_dataset,
    batch_size=2,  # Reduced for spatial reasoning tasks
    collate_fn=data_collator,
    shuffle=True
)

# Optimizer with lower learning rate
optimizer = AdamW(model.parameters(), lr=2e-4)  
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataloader) * 3  # 3 epochs
)

# Training loop
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for epoch in range(1): 
    model.train()
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    
    # Save checkpoint
    accelerator.save_state(output_dir=output_dir)
    print(f"Checkpoint saved after epoch {epoch}")

print("Training completed!")

# =============================================
# Inference on CVBench Dataset (Modified for CLEVR-style evaluation)
# =============================================
print("\nLoading CVBench dataset for inference...")
cvbench_dataset = load_dataset("nyu-visionx/CV-Bench", split="test", cache_dir="/projectnb/cs598/students/achetia")

# Reload best model
model.load_state_dict(torch.load(f"{output_dir}/pytorch_model.bin"))
model.eval()

results_file = f"{output_dir}/cvbench_results_CLEVR.txt"
total_correct = 0

with open(results_file, "w") as f:
    for example in tqdm(cvbench_dataset):
        image = Image.open(example["image_filename"]).convert("RGB")
        question = example["question"]
        
        # Spatial-reasoning focused prompt
        prompt = f"""
        Analyze the spatial relationships in this image carefully.
        Question: {question}
        Answer:"""
        
        inputs = processor(
            text=f"<image>\n{prompt}",
            images=image,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id
            )
        
        pred_answer = processor.decode(outputs[0], skip_special_tokens=True)
        pred_answer = pred_answer.split("Answer:")[-1].strip()
        
        is_correct = pred_answer.lower() == example["answer"].lower()
        total_correct += int(is_correct)
        
        f.write(f"Q: {question}\nGT: {example['answer']}\nPred: {pred_answer}\nCorrect: {is_correct}\n\n")

    accuracy = total_correct / len(cvbench_dataset) * 100
    f.write(f"\nFinal Accuracy: {accuracy:.2f}%")

print(f"Evaluation completed! Accuracy: {accuracy:.2f}%")
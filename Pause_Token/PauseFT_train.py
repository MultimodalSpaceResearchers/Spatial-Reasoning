#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
cwd = os.getcwd()
print("Current Working Directory:", cwd)


# In[ ]:

import copy 
from datasets import load_dataset
import io
import matplotlib.pyplot as plt
import PIL
from PIL import Image

#dataset = load_dataset("array/SAT", batch_size=128, cache_dir="/projectnb/cs598/students/achetia")
# # dataset should have a training and validation key

# example = dataset['train'][15] # example 10th item

# image = example['image'] # this is a list of images. Some questions are on one image, and some on 2 images
# question = example['question']
# answer_choices = example['choices']
# correct_answer = example['answer']

# print(f"Question: {question}")
# for idx, choice in enumerate(answer_choices):
#     print(f"{idx + 1}: {choice}")

# print(f"Correct Answer: {correct_answer}")

# image.show()



# In[ ]:


# %pip install nvidia-pyindex


# In[ ]:


# %pip install nvidia-nccl


# In[ ]:


# !nvidia-smi


# In[ ]:


# !nvcc --version


# In[2]:


import torch
import transformers
import torch.nn as nn
from transformers import LlavaProcessor, AutoModelForCausalLM, TrainingArguments, LlamaTokenizer
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


# In[3]:


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# In[ ]:


# train_dataset = load_dataset("yangfu2/CVBench_Train")
# test_dataset = load_dataset("yangfu2/CVBench_Test")

# print(f"Train dataset size: {len(train_dataset['train'])}")
# print(f"Test dataset size: {len(test_dataset['train'])}")


# In[ ]:


# def process_dataset(dataset, split, output_dir):
#     data = []
#     for item in dataset:
#         buffered = BytesIO()
#         item['image'].save(buffered, format="JPEG")
#         image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
#         conversation = [
#             {"from": "human", "value": f"<image>\n{item['question']}"},
#             {"from": "gpt", "value": item['answer']}
#         ]
#         entry = {
#             "id": f"{split}_{item['idx']}",
#             "image": image_base64,  # Use the Base64 encoded string
#             "conversations": conversation
#         }
#         data.append(entry)
    
#     output_file = os.path.join(output_dir, f"{split}.json")
#     with open(output_file, 'w') as f:
#         json.dump(data, f, indent=2)


# In[ ]:


# output_dir = "cvbench_llava_format"
# os.makedirs(output_dir, exist_ok=True)

# process_dataset(train_dataset['train'], "train", output_dir)
# process_dataset(test_dataset['train'], "test", output_dir)


# In[4]:


# model_path = "deepseek-ai/Janus-Pro-7B"
model_path = "llava-hf/llava-1.5-7b-hf"
cache_dir = "/projectnb/cs598/projects/cool_proj/SuryaWorking/Pause_Finetune/cache"
output_d = "./LLAVA-lora-finetuned"



# In[ ]:


lora_config = LoraConfig(
    r=16,                       
    lora_alpha=32,              
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],#, "gate_proj", "up_proj", "down_proj"
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# In[6]:


print("Loading model and processor...")
#processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)



print("Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=cache_dir)
tokenizer.padding_side = "left"

print("Adding special tokens...")
pause_token = {"additional_special_tokens": ["<pause>"]}
tokenizer.add_special_tokens(pause_token)

# Load model
print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    trust_remote_code=True,
    cache_dir=cache_dir
).to(torch.float16)

# Resize embeddings for the newly added tokens
print("Resizing model embeddings...")
model.resize_token_embeddings(len(tokenizer))


print("Modified Tokenizer")

# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     # Wrap model with DataParallel
#     vl_gpt = nn.DataParallel(vl_gpt, device_ids=[0, 1])

model = model.cuda().eval()


# In[7]:


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


# In[8]:


# model = vl_gpt.to(torch.device("cuda:1"))
print("Preparing model for LoRA fine-tuning...")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# In[9]:


dataset = load_dataset("tau/commonsense_qa", batch_size=128, cache_dir="/projectnb/cs598/projects/cool_proj/SuryaWorking/Pause_Finetune/dataset")
print("Dataset loaded")


# In[10]:


# def preprocess_function(examples):
#     for example in examples:
#         image = [example["image"]]
            
#         question = example["question"]
#         answer_choices = example["choices"]
#         correct_answer = example["answer"]

#         prompt = f"""
#         Instructions: Answer the following question using the given options.
#         Enclose your answer in [].

#         Question: {question}

#         Options: {answer_choices}
#         """

#         response = f"[{correct_answer}]"
        
#         conversations = [
#             {
#                 "role": "<|User|>",
#                 "content": f"<image_placeholder>\n{prompt}",
#                 "images": image,
#             },
#             {"role": "<|Assistant|>", "content": response},
#         ]
    
#     pil_images = load_pil_images(conversations)
#     inputs = vl_chat_processor(
#         conversations=conversations, images=pil_images, force_batchify=True
#     ).to(vl_gpt.device)
    
#     labels = tokenizer(examples["answer"], padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    
#     return {"inputs_embeds": inputs["inputs_embeds"], "labels": labels}
@dataclass
class Collator:
    tokenizer: transformers.PreTrainedTokenizer
    max_length: int = 512

    def __call__(self, examples):
        input_ids = []
        attention_masks = []
        labels = []
        
        for example in examples:
            question = example["question"]
            choices = example["choices"]
            labels_list = choices["label"]
            texts_list = choices["text"]
            correct_answer = example["answerKey"]

            # Join answer choices in prompt
            answer_choices = ", ".join([f"{label}: {text}" for label, text in zip(labels_list, texts_list)])

            # Format prompt (same as finetuning)
            prompt = f"""
            You have to answer the following question using the given options.
            <pause>
            Question: {question}
            <pause>
            Your answer must contain just the options given below.
            Options: {answer_choices}
            <pause>
            Enclose the answer in brackets like [answer]. <pause>
            peepeep
            """

            # Match what you expect at inference
            correct_text = texts_list[labels_list.index(correct_answer)]
            target = f"[{correct_text}]"
            

            # Tokenize prompt and response
            prompt_tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=False
            )
            target_tokens = self.tokenizer(
                target,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=False
            )

            # Mask out padding in target
            target_ids = target_tokens.input_ids[0]
            target_ids = torch.where(
                target_ids == self.tokenizer.pad_token_id,
                torch.full_like(target_ids, -100),
                target_ids,
            )

            input_ids.append(prompt_tokens.input_ids[0])
            attention_masks.append(prompt_tokens.attention_mask[0])
            labels.append(target_ids)

        # Stack everything
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels)
        }
    



# In[11]:


data_collator = Collator(tokenizer = tokenizer, max_length=512)
accelerator = Accelerator()


# In[ ]:


# training_args = TrainingArguments(
#     output_dir=output_dir,
#     num_train_epochs=3,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     learning_rate=2e-4,
#     warmup_steps=100,
#     logging_steps=10,
#     save_steps=200,
#     fp16=True,  # Use mixed precision training
#     optim="adamw_torch",
#     report_to="tensorboard",
#     remove_unused_columns=False,
#     local_rank=-1,  # for distributed training
#     ddp_find_unused_parameters=False,
# )


# In[ ]:


# trainer = transformers.Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["validation"] if "validation" in dataset else None,
#     data_collator=data_collator,
# )


# In[ ]:


# print("Starting training...")
# trainer.train()


# In[12]:


model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, 
    AdamW(model.parameters(), lr=5e-5),
    torch.utils.data.DataLoader(dataset["train"], batch_size=1, collate_fn=data_collator),
    torch.utils.data.DataLoader(dataset["validation"], batch_size=1, collate_fn=data_collator) if "validation" in dataset else None
)

batch = next(iter(train_dataloader))

decoded_prompt = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
decoded_label = tokenizer.decode(
    [token_id for token_id in batch["labels"][0] if token_id != -100],
    skip_special_tokens=False
)

print("==== DEBUG: PROMPT ====")
print(decoded_prompt)

print("\n==== DEBUG: LABEL (no -100) ====")
print(decoded_label)

print("\n==== DEBUG: Raw shapes ====")
print(f"input_ids shape: {batch['input_ids'].shape}")
print(f"attention_mask shape: {batch['attention_mask'].shape}")
print(f"labels shape: {batch['labels'].shape}")

print("Pad token ID:", tokenizer.pad_token_id)


# In[13]:


num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = 3 * num_update_steps_per_epoch
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)


# In[ ]:


save_dir = "./LLAVA-lora-finetuned"
os.makedirs(save_dir, exist_ok=True)
EPOCHS = 3
for epoch in range(EPOCHS):
    epoch_dir = f"./LLAVA-lora-finetuned/epoch_{epoch}"
    os.makedirs(epoch_dir, exist_ok=True)
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

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)

print("Training completed")


# In[ ]:





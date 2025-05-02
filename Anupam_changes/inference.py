import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# Load base model and adapter
base_model = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir = "/projectnb/cs598/students/achetia/model/LlaVa"
)
state_dict = torch.load("./LLAVA-lora-finetuned/epoch_0.pt")
model.load_state_dict(state_dict, strict=False)

# Load processor from original model
processor = AutoProcessor.from_pretrained(base_model)

# Load CVBench dataset
cvbench = load_dataset("nyu-visionx/CV-Bench", split="test", cache_dir = "/projectnb/cs598/students/achetia")

# Inference loop
results = []
for example in tqdm(cvbench):
    image = example["image"]
    prompt = f"USER: <image> Choose the correct option:\n{example['prompt']}\nASSISTANT:"
    
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            #pad_token_id=processor.tokenizer.pad_token_id
        )

    response = processor.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1].split()[0]

    results.append({
        "question": example["question"],
        "predicted": response,
        "ground_truth": example["answer"]
    })

# Calculate accuracy
correct = sum(1 for r in results if r["predicted"].lower() == r["ground_truth"].lower())
accuracy = correct / len(results) * 100
print(f"Accuracy: {accuracy:.2f}%")

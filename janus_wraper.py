import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union


@torch.inference_mode()
def generate(input_ids,
             vl_chat_processor,
             vl_gpt,
             width=384,
             height=384,
             temperature=1.0,
             num_images=5,
             cfg_weight=5.0,
             image_token_num_per_image=576,
             patch_size=16, 
             device='cuda'):
    """
    Generate image tokens and visual patches from the given input_ids.
    
    Parameters:
      input_ids (torch.LongTensor): The encoded prompt.
      width (int): Target width of the generated image (default 384).
      height (int): Target height of the generated image (default 384).
      temperature (float): Sampling temperature for generation (default 1.0).
      num_images (int): Number of image variations to generate (default 5).
      cfg_weight (float): Classifier-Free Guidance weight (default 5.0).
      image_token_num_per_image (int): Number of image tokens to generate per image (default 576).
      patch_size (int): Size of a single image patch (default 16).
      
    Returns:
      generated_tokens (torch.Tensor): The generated image tokens.
      patches (torch.Tensor): The decoded image patches.
    """
    # Clear CUDA cache to help avoid out-of-memory errors.
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Create a tokens tensor with shape (num_images * 2, sequence_length)
    # The model expects paired tokens for unconditional and conditional generation.
    tokens = torch.zeros((num_images * 2, len(input_ids)), dtype=torch.int).to(device)
    for i in range(num_images * 2):
        tokens[i, :] = input_ids
        # For every odd-indexed row (the unconditional branch), replace
        # all tokens except the first and last with the pad token.
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    # Get the input embeddings for the tokens.
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    
    # Prepare a tensor to hold the generated tokens.
    generated_tokens = torch.zeros((num_images, image_token_num_per_image), dtype=torch.int).to(device)
    pkv = None  # Past key values for caching.
    
    # Iteratively generate tokens for the image.
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=pkv
            )
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            # Get logits from the generation head.
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            # Split logits into conditional and unconditional parts.
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            # Apply classifier-free guidance.
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            # Convert logits to probabilities.
            probs = torch.softmax(logits / temperature, dim=-1)
            # Sample the next token.
            next_token = torch.multinomial(probs, num_samples=1)
            # Save the token for each image.
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            # Duplicate the sampled token for both branches and reshape.
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            # Prepare new input embeddings for the next iteration.
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
    
    # Decode the generated tokens into image patches.
    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[num_images, 8, width // patch_size, height // patch_size]
    )
    
    return generated_tokens.to(dtype=torch.int), patches

def set_seed(seed: int, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Set random seeds for reproducibility.
    """
    if device == 'cuda':
        torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


def generate_text_output(
    vl_chat_processor,
    vl_gpt,
    vl_tokenizer,
    input_text: str,
    input_images,
    device: str,
    seed: int,
    top_p: float,
    temperature: float,
):
    """
    Generate text output given optional image context.
    Returns: str
    """
    if input_text is None:
        raise ValueError("Text input is required for text output.")

    # Seed environment for reproducible results
    if seed is not None: set_seed(seed)

    # If an image is provided, we do multimodal text generation
    if input_images is not None:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{input_text}",
                "images": input_images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # Convert numpy array to PIL if needed
        # pil_images = [Image.fromarray(input_images)] if isinstance(input_images, np.ndarray) else input_images
        pil_images = [Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image for input_image in input_images]
        prep = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
        t_dtype = torch.bfloat16 if device=='cuda' or device=='mps' else torch.float16
        prep = prep.to(device, dtype=t_dtype)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prep)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prep.attention_mask,
            pad_token_id=vl_tokenizer.eos_token_id,
            bos_token_id=vl_tokenizer.bos_token_id,
            eos_token_id=vl_tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=(temperature != 0),
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
        )
        return vl_tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    else:
        # conversation = [
        #     {
        #         "role": "<|User|>",
        #         "content": f"<image_placeholder>\n{input_text}",
        #         'images': [],
        #     },
        #     {"role": "<|Assistant|>", "content": ""},
        # ]
        # Convert numpy array to PIL if needed
        # prep = vl_chat_processor(conversations=conversation, force_batchify=True)
        # t_dtype = torch.bfloat16 if device=='cuda' or device=='mps' else torch.float16
        # prep = prep.to(device, dtype=t_dtype)
        #
        # inputs_embeds = vl_gpt.prepare_inputs_embeds(**prep)
        # Pure text generation
        # inputs = vl_tokenizer.encode(input_text, return_tensors="pt").to(device)
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{input_text}",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        prep = vl_chat_processor(conversations=conversation, force_batchify=True).to(device)
        t_dtype = torch.bfloat16 if device=='cuda' or device=='mps' else torch.float16
        inputs_embeds = vl_gpt.prepare_inputs_embeds(prep)
        # inputs = vl_tokenizer.encode(input_text, return_tensors="pt").to(device)
        outputs = vl_gpt.language_model.generate(
            # inputs_embeds=inputs_embeds,
            # inputs=inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        return vl_tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


def generate_image_output(
    vl_chat_processor,
    vl_gpt,
    vl_tokenizer,
    input_text: str,
    input_images,
    device: str,
    seed: int,
    guidance: float,
    t2i_temperature: float,
    num_images: int,
    width: int,
    height: int,
    out_width: int,
    out_height: int,
    image_token_num_per_image: int,
    patch_size: int,
):
    """
    Generate images conditioned on text (and optionally on an image).
    Returns: list of PIL.Image
    """
    if input_text is None:
        raise ValueError("Text prompt is required for image generation.")

    # Seed environment for reproducible results
    if seed is not None: set_seed(seed)

    with torch.inference_mode():
        # Build the prompt
        if input_images is not None:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{input_text}",
                    "images": input_images,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            prep = vl_chat_processor(conversations=conversation, images=input_images, force_batchify=True)
            prompt = prep.sft_format[0] + vl_chat_processor.image_start_tag
        else:
            messages = [
                {"role": "<|User|>", "content": input_text},
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=messages,
                sft_format=vl_chat_processor.sft_format,
                system_prompt=""
            )
            prompt = sft_format + vl_chat_processor.image_start_tag

        input_ids = torch.LongTensor(vl_tokenizer.encode(prompt)).to(device)

        # Prepare unconditional/conditional tokens
        tokens = torch.zeros((num_images * 2, len(input_ids)), dtype=torch.int).to(device)
        for i in range(num_images * 2):
            tokens[i, :] = input_ids
            # Unconditional branch: pad the middle tokens
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id
        
        # Convert tokens to embeddings
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

        # Generate tokens for the image patches
        generated_tokens = torch.zeros((num_images, image_token_num_per_image), dtype=torch.int).to(device)
        pkv = None
        for i in range(image_token_num_per_image):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=pkv
            )
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            # Classifier-free guidance
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + guidance * (logit_cond - logit_uncond)

            # Sample next token
            probs = torch.softmax(logits / t2i_temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            # Prepare next input embeddings
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        
        # Convert tokens to image patches
        patches = vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(torch.int),
            shape=[num_images, 8, width // patch_size, height // patch_size]
        )
        patches = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        patches = np.clip((patches + 1) / 2 * 255, 0, 255).astype(np.uint8)

        # Resize images to desired resolution
        out_images = [
            Image.fromarray(patches[i]).resize((out_width, out_height), Image.LANCZOS)
            for i in range(num_images)
        ]
        return out_images


def janus_pro_generate(
    vl_chat_processor,
    vl_gpt,
    input_text=None,
    input_images=None,
    device='cuda',
    output_mode="text",
    seed=None,
    top_p=0.95,
    temperature=0.1,
    guidance=5,
    t2i_temperature=1.0,
    num_images=5,
    width=384,
    height=384,
    out_width=768,
    out_height=768,
    image_token_num_per_image=576,
    patch_size=16,
    use_coconut=False,
    num_continuous_thoughts=3,
):
    """
    Unified entry point for Janus-Pro that dispatches to text or image generation.
    
    Parameters:
        use_coconut (bool): Whether to use the Coconut (Chain of Continuous Thought) approach
        num_continuous_thoughts (int): Number of continuous thoughts to use when use_coconut=True
    """
    if seed is None:
        seed = np.random.randint(1)
    
    if use_coconut and output_mode == "text":
        return generate_text_with_coconut(
            vl_chat_processor,
            vl_gpt,
            vl_chat_processor.tokenizer,
            input_text=input_text,
            input_images=input_images,
            device=device,
            seed=seed,
            top_p=top_p,
            temperature=temperature,
            num_continuous_thoughts=num_continuous_thoughts
        )
    elif output_mode == "text":
        return generate_text_output(
            vl_chat_processor,
            vl_gpt,
            vl_chat_processor.tokenizer,
            input_text=input_text,
            input_images=input_images,
            device=device,
            seed=seed,
            top_p=top_p,
            temperature=temperature
        )
    elif output_mode == "image":
        return generate_image_output(
            vl_chat_processor,
            vl_gpt,
            vl_chat_processor.tokenizer,
            input_text=input_text,
            input_images=input_images,
            device=device,
            seed=seed,
            guidance=guidance,
            t2i_temperature=t2i_temperature,
            num_images=num_images,
            width=width,
            height=height,
            out_width=out_width,
            out_height=out_height,
            image_token_num_per_image=image_token_num_per_image,
            patch_size=patch_size
        )
    else:
        raise ValueError("output_mode must be either 'text' or 'image'.")

def generate_text_with_coconut(
    vl_chat_processor,
    vl_gpt,
    vl_tokenizer,
    input_text: str,
    input_images,
    device: str,
    seed: int,
    top_p: float,
    temperature: float,
    num_continuous_thoughts: int = 3,
):
    """
    Generate text output using the Coconut (Chain of Continuous Thought) approach.
    
    This implementation uses continuous latent space reasoning instead of token-by-token
    generation for intermediate reasoning steps. The model's last hidden states are used
    directly as input embeddings for subsequent reasoning steps.
    
    Parameters:
        vl_chat_processor: The vision-language chat processor
        vl_gpt: The vision-language model
        vl_tokenizer: The tokenizer for the language model
        input_text: The text prompt
        input_image: Optional image input
        device: The device to run inference on
        seed: Random seed for reproducibility
        top_p: Top-p sampling parameter
        temperature: Temperature for sampling
        num_continuous_thoughts: Number of continuous thought steps to perform
        
    Returns:
        str: The generated text response
    """
    if input_text is None:
        raise ValueError("Text input is required for text output.")

    # Seed environment for reproducible results
    if seed is not None: set_seed(seed)
    
    # Special tokens for continuous thought
    BOT_TOKEN = "<thought>"  # Beginning of thought
    EOT_TOKEN = "</thought>"  # End of thought
    
    # Add special tokens if they don't exist
    special_tokens = {"additional_special_tokens": [BOT_TOKEN, EOT_TOKEN]}
    if BOT_TOKEN not in vl_tokenizer.get_vocab() or EOT_TOKEN not in vl_tokenizer.get_vocab():
        vl_tokenizer.add_special_tokens(special_tokens)
        # Resize token embeddings to accommodate new special tokens
        vl_gpt.resize_token_embeddings(len(vl_tokenizer))
    
    # Get token IDs for special tokens
    bot_token_id = vl_tokenizer.convert_tokens_to_ids(BOT_TOKEN)
    eot_token_id = vl_tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    
    # Prepare the input
    if input_images is not None:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{input_text}",
                "images": input_images,
            },
            {"role": "<|Assistant|>", "content": f"{BOT_TOKEN}"},
        ]
        # Convert numpy array to PIL if needed
        pil_images = [Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image for input_image in input_images ]
        prep = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
    else:
        # Text-only input
        messages = [
            {"role": "<|User|>", "content": input_text},
            {"role": "<|Assistant|>", "content": f"{BOT_TOKEN}"},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=""
        )
        # Tokenize the input
        inputs = vl_tokenizer.encode(sft_format, return_tensors="pt").to(device)
        # Create attention mask
        attention_mask = torch.ones_like(inputs)
        prep = type('obj', (), {
            'input_ids': inputs,
            'attention_mask': attention_mask
        })
    
    # Set data type based on device
    t_dtype = torch.bfloat16 if device=='cuda' or device=='mps' else torch.float16
    prep = prep.to(device, dtype=t_dtype)
    
    # Get input embeddings
    if hasattr(prep, 'input_ids'):
        inputs_embeds = vl_gpt.get_input_embeddings()(prep.input_ids)
    else:
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prep)
    
    # Initialize past key values for caching
    past_key_values = None
    
    # Perform continuous thought reasoning
    with torch.inference_mode():
        # First forward pass to get to the BOT token
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=prep.attention_mask,
            use_cache=True,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
        
        # Get the last hidden state (continuous thought)
        last_hidden_state = outputs.last_hidden_state[:, -1:, :]
        
        # Chain of continuous thoughts
        for i in range(num_continuous_thoughts):
            # Use the last hidden state directly as the next input embedding
            continuous_thought_embed = last_hidden_state
            
            # Update attention mask for the new token
            new_attention_mask = torch.cat([
                prep.attention_mask, 
                torch.ones((prep.attention_mask.shape[0], 1), device=device)
            ], dim=1)
            
            # Forward pass with the continuous thought
            outputs = vl_gpt.language_model.model(
                inputs_embeds=continuous_thought_embed,
                attention_mask=new_attention_mask,
                use_cache=True,
                past_key_values=past_key_values
            )
            
            # Update past key values and last hidden state
            past_key_values = outputs.past_key_values
            last_hidden_state = outputs.last_hidden_state
            
            # Update attention mask for next iteration
            prep.attention_mask = new_attention_mask
    
    # Add EOT token embedding to signal end of continuous thoughts
    eot_token_tensor = torch.tensor([[eot_token_id]], device=device)
    eot_embedding = vl_gpt.get_input_embeddings()(eot_token_tensor)
    
    # Update attention mask for EOT token
    final_attention_mask = torch.cat([
        prep.attention_mask, 
        torch.ones((prep.attention_mask.shape[0], 1), device=device)
    ], dim=1)
    
    # Generate the final answer after continuous thoughts
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=eot_embedding,
        attention_mask=final_attention_mask,
        pad_token_id=vl_tokenizer.eos_token_id,
        bos_token_id=vl_tokenizer.bos_token_id,
        eos_token_id=vl_tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=(temperature != 0),
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        past_key_values=past_key_values
    )
    
    # Decode the output, skipping special tokens
    generated_text = vl_tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    return generated_text

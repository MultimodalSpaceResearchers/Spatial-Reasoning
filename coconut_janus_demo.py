#%% md
# # Demonstrating Coconut Approach with Janus Model
# 
# This notebook demonstrates how to use the Chain of Continuous Thought (Coconut) approach with the Janus model. The Coconut approach allows the model to reason in a continuous latent space rather than being restricted to token-by-token reasoning in language space.
#%% md
# ## Setup and Imports
# 
# First, let's import the necessary libraries and load the Janus model.
#%%
from janus_wraper import *
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
model_path = '/Users/nover/models/deepseek-ai/Janus-Pro-7B'

# specify the path to the model
# model_path = "/projectnb/cs598/projects/cool_proj/model"

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
device = 'cpu'

chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
)
gpt = gpt.to(torch.bfloat16).to(device).eval()


#%% md
# ## Understanding the Coconut Approach
# 
# The Coconut approach (Chain of Continuous Thought) allows the model to reason in a continuous latent space rather than being restricted to token-by-token reasoning in language space. This approach:
# 
# 1. Uses the last hidden state of the model as a representation of the reasoning state ("continuous thought")
# 2. Feeds this continuous thought directly back to the model as the next input embedding
# 3. Allows the model to perform multiple reasoning steps in the latent space before generating a final answer
# 
# This approach has several advantages:
# - The continuous thought can encode multiple alternative reasoning paths simultaneously
# - It enables breadth-first search (BFS) reasoning patterns
# - It can be more efficient, requiring fewer tokens for complex reasoning tasks
#%% md
# ## Example 1: Text-Only Reasoning with Coconut
#%%
from janus_wraper import janus_pro_generate
from PIL import Image

# Define a complex reasoning question
complex_question = "If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?"

image = Image.open("img.jpg")

# Generate response using standard approach
standard_response = janus_pro_generate(
    chat_processor,
    gpt,
    device=device,
    input_text=complex_question,
    input_images=[image],
    output_mode="text",
    use_coconut=False,
)

print("Standard Response:")
print(standard_response)
#%%
# Generate response using Coconut approach
coconut_response = janus_pro_generate(
    chat_processor,
    gpt,
    input_text=complex_question,
    input_images=[image],
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_mode="text",
    temperature=0.1,
    use_coconut=True,  # Enable continuous thought reasoning
    num_continuous_thoughts=3  # Use 3 continuous thought steps
)

print("Coconut Response:")
print(coconut_response)
#%% md
# ## Example 2: Visual Reasoning with Coconut
# 
# Now let's try a visual reasoning task using an image.
#%%
# Load an example image
image_path = "img.jpg"  # Replace with your image path
image = Image.open(image_path)

# Display the image
image.show()
#%%
# Define a complex visual reasoning question
visual_question = "What objects are in this image and how are they arranged? Explain the spatial relationships."

# Generate response using standard approach
standard_visual_response = janus_pro_generate(
    vl_chat_processor,
    vl_gpt,
    input_text=visual_question,
    input_image=image,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_mode="text",
    temperature=0.1,
    use_coconut=False
)

print("Standard Visual Response:")
print(standard_visual_response)
#%%
# Generate response using Coconut approach
coconut_visual_response = janus_pro_generate(
    vl_chat_processor,
    vl_gpt,
    input_text=visual_question,
    input_image=image,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_mode="text",
    temperature=0.1,
    use_coconut=True,
    num_continuous_thoughts=4  # Use 4 continuous thought steps for more complex visual reasoning
)

print("Coconut Visual Response:")
print(coconut_visual_response)
#%% md
# ## Example 3: Logical Reasoning with Coconut
# 
# Let's try a logical reasoning problem that requires planning and backtracking.
#%%
# Define a logical reasoning problem
logical_problem = """
Every grimpus is a yimpus. Every worpus is a jelpus. Every zhorpus is a sterpus. 
Alex is a grimpus. Every lumpus is a yumpus. 
Question: Is Alex a gorpus or bompus?
"""

# Generate response using standard approach
standard_logical_response = janus_pro_generate(
    vl_chat_processor,
    vl_gpt,
    input_text=logical_problem,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_mode="text",
    temperature=0.1,
    use_coconut=False
)

print("Standard Logical Response:")
print(standard_logical_response)
#%%
# Generate response using Coconut approach
coconut_logical_response = janus_pro_generate(
    vl_chat_processor,
    vl_gpt,
    input_text=logical_problem,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_mode="text",
    temperature=0.1,
    use_coconut=True,
    num_continuous_thoughts=5  # Use more continuous thoughts for complex logical reasoning
)

print("Coconut Logical Response:")
print(coconut_logical_response)
#%% md
# ## Experimenting with Different Numbers of Continuous Thoughts
# 
# Let's see how the number of continuous thoughts affects the reasoning process.
#%%
# Define a complex math problem
math_problem = "If a rectangle has a length of 12 cm and a width of 8 cm, what is its area and perimeter?"

# Try with different numbers of continuous thoughts
for num_thoughts in [1, 2, 3, 5]:
    response = janus_pro_generate(
        vl_chat_processor,
        vl_gpt,
        input_text=math_problem,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_mode="text",
        temperature=0.1,
        use_coconut=True,
        num_continuous_thoughts=num_thoughts
    )
    
    print(f"\nResponse with {num_thoughts} continuous thoughts:")
    print(response)
#%% md
# ## Conclusion
# 
# The Coconut approach allows the Janus model to reason in a continuous latent space, which can lead to more effective reasoning for complex problems. By using continuous thoughts, the model can:
# 
# 1. Encode multiple potential reasoning paths simultaneously
# 2. Perform breadth-first search-like reasoning
# 3. Potentially provide more accurate answers for problems that require planning and backtracking
# 
# This approach is particularly effective for logical reasoning tasks and other problems that benefit from exploring multiple reasoning paths before committing to a final answer.
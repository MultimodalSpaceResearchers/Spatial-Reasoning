 # prior work
0. 
 
    Q1 Prior work on using latent reasoning 
    ---
 1. [Implicit Chain of Thought Reasoning via Knowledge Distillation](https://arxiv.org/pdf/2311.01460): This work demonstrates the basic idea of what I am talking about applying to this project. However, it applies it to narrow mathematical tasks, and not spatial reasoning. 
 2. [Calibrating Reasoning in Language Models with Internal Consistency](https://arxiv.org/abs/2405.18711): This work is on using the constancy in latent representation as the input is generated (per token?) to measure the confidence of the model. This indicates that the latent space representations may be a beter representation of model informatin. 
 3. [Human symbol manipulation within an integrated cognitive architecture](https://pubmed.ncbi.nlm.nih.gov/21702777/): Cited form Implicit.. they note that this is how we reason in the human mind, internally. Ie. we don't necessarily verbalize our thought processes. 
 4. [LaRS: Latent Reasoning Skills for Chain-of-Thought Reasoning](https://arxiv.org/pdf/2312.04684): This seems very similar, however they add layers(?) and train these added layers.
 5. [Chain of Continuous Thoughts](https://benjamincongdon.me/blog/2024/12/14/Chain-of-Continuous-Thoughts/): COCONUT approach this approach removes only the embeddings. It seems very similar, however they don't evaluate the model's spatial reasoning abilities, and only remove 1 layer. 
 6. [Neuro-symbolic Training for Reasoning over Spatial Language](https://arxiv.org/abs/2406.13828) 
 7. [Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation](https://openreview.net/forum?id=jxo70B9fQo)
 
    Q2 Reasoning Enhancements 
    ---
 8. [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) another approach, using trees. 
 9. [Mind's Eye of LLMs: Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models](https://arxiv.org/abs/2404.03622): This involves a technique of prompting the llm to visualize it's thought process. 
 10. [Whiteboard-of-Thought: Thinking Step-by-Step Across Modalities](https://arxiv.org/abs/2406.14562) This is a method of chain of thought that passes an image in the latent space. ie a whiteboard.

# Model
1. janus pro: [Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling](https://arxiv.org/pdf/2501.17811)
2. deepseek's llama 3.3 70B distill: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)
3. [openbmb/MiniCPM-o-2_6 ](https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9)

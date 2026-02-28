# Qwen3 Prompt Pro - InvokeAI Node

An InvokeAI community node that uses a Qwen3 encoder model as a causal LLM to generate enhanced image prompts, then encodes them into conditioning embeddings for both **Z-Image** and **Flux Klein** pipelines.

## Features

- **Prompt Enhancement** - Uses Qwen3 as a causal LLM to expand brief descriptions into detailed, vivid image generation prompts
- **Dual Pipeline Output** - Produces ready-to-use conditioning for both Z-Image and Flux Klein in a single node
- **Reuses Loaded Models** - Works with the same Qwen3 encoder already loaded for Z-Image or Flux2 Klein, no extra model needed
- **Configurable Generation** - Adjustable temperature, top-p, max tokens, and custom system prompts
- **Thinking Mode** - Optional Qwen3 thinking mode for potentially higher quality prompt enhancement
- **LoRA Support** - Applies LoRAs to the text encoder during the encoding step

## Installation

Place this folder into your InvokeAI `nodes` directory:

```
invokeai/nodes/qwen3_prompt_pro/
```

The node will be automatically discovered on the next InvokeAI startup.

## Usage

1. Load a Qwen3 encoder model (the same one used by Z-Image or Flux2 Klein text encoder nodes)
2. Connect it to the **Qwen3 Prompt Pro** node
3. Enter a brief prompt
4. The node outputs:
   - **Enhanced Prompt** - The LLM-generated detailed prompt text
   - **Z-Image Conditioning** - Embeddings ready for Z-Image pipelines
   - **Flux Conditioning** - Embeddings ready for Flux Klein pipelines

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | `""` | Input text prompt to enhance |
| `system_prompt` | *(built-in)* | System prompt guiding the enhancement style |
| `max_tokens` | `300` | Maximum tokens to generate (1-2048) |
| `temperature` | `0.7` | Sampling temperature (0.0-2.0) |
| `top_p` | `0.9` | Nucleus sampling threshold (0.0-1.0) |
| `enable_thinking` | `false` | Enable Qwen3 thinking mode |
| `mask` | `None` | Optional region mask for conditioning |

## Requirements

- Tested with InvokeAI v6.11.1 with Z-Image / Flux2 Klein support
- A compatible Qwen3 encoder model

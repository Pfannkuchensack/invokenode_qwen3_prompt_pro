"""Qwen3 Prompt Pro Node.

Uses a Qwen3 encoder model (already loaded for Z-Image or Flux2 Klein) as a causal LLM
to generate an enhanced prompt, then encodes it into conditioning embeddings for both
Z-Image and Flux Klein pipelines.
"""

import re
from contextlib import ExitStack
from typing import Iterator, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    OutputField,
    TensorField,
    UIComponent,
    ZImageConditioningField,
)
from invokeai.app.invocations.flux2_klein_text_encoder import KLEIN_EXTRACTION_LAYERS
from invokeai.app.invocations.model import Qwen3EncoderField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_T5_PREFIX
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_QWEN3_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    FLUXConditioningInfo,
    ZImageConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert prompt writer for AI image generation. "
    "Given a brief description, expand it into a detailed, vivid prompt suitable for generating high-quality images. "
    "Only output the expanded prompt, nothing else."
)

# Max sequence length for encoding (matches Z-Image and Klein defaults)
_MAX_SEQ_LEN = 512

# Regex to strip Qwen3 thinking blocks from generated output
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@invocation_output("qwen3_prompt_pro_output")
class Qwen3PromptProOutput(BaseInvocationOutput):
    """Output containing the enhanced prompt text and conditioning for Z-Image and Flux Klein."""

    enhanced_prompt: str = OutputField(description="The enhanced prompt text")
    z_image_conditioning: ZImageConditioningField = OutputField(description="Z-Image conditioning embeddings")
    flux_conditioning: FluxConditioningField = OutputField(description="Flux Klein conditioning embeddings")


@invocation(
    "qwen3_prompt_pro",
    title="Qwen3 Prompt Pro",
    tags=["llm", "text", "prompt", "qwen3"],
    category="prompt",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Qwen3PromptProInvocation(BaseInvocation):
    """Use a Qwen3 encoder as a causal LLM to enhance a prompt, then encode it for Z-Image and Flux Klein.

    Accepts the same Qwen3 encoder model used by Z-Image and Flux2 Klein text encoder nodes.
    Outputs the enhanced prompt as text plus ready-to-use conditioning for both pipelines.
    """

    prompt: str = InputField(default="", description="Input text prompt.", ui_component=UIComponent.Textarea)
    system_prompt: str = InputField(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt that guides prompt enhancement. Customizable.",
        ui_component=UIComponent.Textarea,
    )
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that the conditioning applies to.",
    )
    max_tokens: int = InputField(
        default=300,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = InputField(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher = more creative, lower = more focused.",
    )
    top_p: float = InputField(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold.",
    )
    enable_thinking: bool = InputField(
        default=False,
        description="Enable Qwen3 thinking mode during generation. Slower but potentially higher quality.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Qwen3PromptProOutput:
        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (cached_weights, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            if not isinstance(text_encoder, PreTrainedModel):
                raise TypeError(
                    f"Expected PreTrainedModel for text encoder, got {type(text_encoder).__name__}."
                )
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(
                    f"Expected PreTrainedTokenizerBase for tokenizer, got {type(tokenizer).__name__}."
                )

            # Use the actual compute device (CUDA if available).
            # model.device can report CPU for GGUF models with CPU-resident embeddings,
            # even though decoder layers are on GPU. Move the whole model to be safe.
            device = TorchDevice.choose_torch_device()
            text_encoder.to(device)

            # --- Step 1: Generate enhanced prompt (no LoRAs) ---
            context.util.signal_progress("Generating enhanced prompt with Qwen3")
            enhanced_prompt = self._generate_prompt(text_encoder, tokenizer, device)
            context.logger.info(f"Qwen3 Prompt Pro enhanced: {enhanced_prompt[:200]}...")

            # --- Step 2: Encode for Z-Image (with LoRAs) ---
            context.util.signal_progress("Encoding enhanced prompt for Z-Image")
            lora_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
            with LayerPatcher.apply_smart_model_patches(
                model=text_encoder,
                patches=self._lora_iterator(context),
                prefix=Z_IMAGE_LORA_QWEN3_PREFIX,
                dtype=lora_dtype,
                cached_weights=cached_weights,
            ):
                z_image_embeds = self._encode_z_image(enhanced_prompt, text_encoder, tokenizer, device, context)

            z_image_embeds = z_image_embeds.detach().to("cpu")
            z_conditioning_data = ConditioningFieldData(
                conditionings=[ZImageConditioningInfo(prompt_embeds=z_image_embeds)]
            )
            z_conditioning_name = context.conditioning.save(z_conditioning_data)

            # --- Step 3: Encode for Flux Klein (with LoRAs) ---
            context.util.signal_progress("Encoding enhanced prompt for Flux Klein")
            with LayerPatcher.apply_smart_model_patches(
                model=text_encoder,
                patches=self._lora_iterator(context),
                prefix=FLUX_LORA_T5_PREFIX,
                dtype=lora_dtype,
                cached_weights=cached_weights,
            ):
                klein_embeds, klein_pooled = self._encode_flux_klein(
                    enhanced_prompt, text_encoder, tokenizer, device, context
                )

            klein_embeds = klein_embeds.detach().to("cpu")
            klein_pooled = klein_pooled.detach().to("cpu")
            flux_conditioning_data = ConditioningFieldData(
                conditionings=[FLUXConditioningInfo(clip_embeds=klein_pooled, t5_embeds=klein_embeds)]
            )
            flux_conditioning_name = context.conditioning.save(flux_conditioning_data)

        return Qwen3PromptProOutput(
            enhanced_prompt=enhanced_prompt,
            z_image_conditioning=ZImageConditioningField(
                conditioning_name=z_conditioning_name, mask=self.mask
            ),
            flux_conditioning=FluxConditioningField(
                conditioning_name=flux_conditioning_name, mask=self.mask
            ),
        )

    def _generate_prompt(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> str:
        """Generate an enhanced prompt using Qwen3 as a causal LLM."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})

        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            formatted_prompt: str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        else:
            if self.system_prompt:
                formatted_prompt = f"{self.system_prompt}\n\nUser: {self.prompt}\nAssistant:"
            else:
                formatted_prompt = self.prompt

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device=device)
        input_length = inputs["input_ids"].shape[1]

        output = model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        generated_tokens = output[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Strip thinking blocks if present
        if self.enable_thinking:
            response = _THINK_RE.sub("", response).strip()

        return response

    def _encode_z_image(
        self,
        prompt: str,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        context: InvocationContext,
    ) -> torch.Tensor:
        """Encode prompt for Z-Image. Matches ZImageTextEncoderInvocation._encode_prompt."""
        # Apply chat template with enable_thinking=True (Z-Image default)
        try:
            prompt_formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except (AttributeError, TypeError) as e:
            context.logger.warning(f"Chat template failed ({e}), using raw prompt.")
            prompt_formatted = prompt

        text_inputs = tokenizer(
            prompt_formatted,
            padding="max_length",
            max_length=_MAX_SEQ_LEN,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        assert isinstance(text_input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)

        prompt_mask = attention_mask.to(device).bool()
        outputs = text_encoder(
            text_input_ids.to(device),
            attention_mask=prompt_mask,
            output_hidden_states=True,
        )

        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError("Text encoder did not return hidden_states.")
        if len(outputs.hidden_states) < 2:
            raise RuntimeError(
                f"Expected at least 2 hidden states, got {len(outputs.hidden_states)}."
            )

        # Second-to-last hidden state, filtered by attention mask (Z-Image convention)
        prompt_embeds = outputs.hidden_states[-2]
        prompt_embeds = prompt_embeds[0][prompt_mask[0]]

        return prompt_embeds

    def _encode_flux_klein(
        self,
        prompt: str,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        context: InvocationContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt for Flux Klein. Matches Flux2KleinTextEncoderInvocation._encode_prompt."""
        # Apply chat template with enable_thinking=False (Klein default)
        text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=_MAX_SEQ_LEN,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError("Text encoder did not return hidden_states.")

        num_hidden_layers = len(outputs.hidden_states)

        # Extract and stack hidden states from Klein extraction layers (9, 18, 27)
        hidden_states_list = []
        for layer_idx in KLEIN_EXTRACTION_LAYERS:
            if layer_idx >= num_hidden_layers:
                layer_idx = num_hidden_layers - 1
            hidden_states_list.append(outputs.hidden_states[layer_idx])

        out = torch.stack(hidden_states_list, dim=1)
        out = out.to(dtype=text_encoder.dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        # Mean-pool last hidden state for pooled embeddings
        last_hidden_state = outputs.hidden_states[-1]
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        sum_embeds = (last_hidden_state * expanded_mask).sum(dim=1)
        num_tokens = expanded_mask.sum(dim=1).clamp(min=1)
        pooled_embeds = sum_embeds / num_tokens

        return prompt_embeds, pooled_embeds

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models to apply to the Qwen3 text encoder."""
        for lora in self.qwen3_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}."
                )
            yield (lora_info.model, lora.weight)
            del lora_info

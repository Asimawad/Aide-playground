# START OF MODIFIED FILE aide-ds/aide/backend/backend_local.py
import logging
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from funcy import notnone, select_values

from aide.backend.utils import FunctionSpec, OutputType, opt_messages_to_list

logger = logging.getLogger("aide")

# Using a dictionary cache to support multiple models concurrently
class LocalLLMCache:
    _models_tokenizers = {}
    _pipelines = {} # Cache for pipelines if needed later

    @classmethod
    def get_model_tokenizer(cls, model_name: str, load_in_4bit: bool = True):
        """Loads model and tokenizer, caching them."""
        if model_name not in cls._models_tokenizers:
            logger.info(f"Loading local model and tokenizer: {model_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    # Common practice: set pad token to eos token if not defined
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Tokenizer pad_token set to eos_token: {tokenizer.eos_token}")

                quantization_config = None
                if load_in_4bit and torch.cuda.is_available():
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16, # Changed to bfloat16 for better compatibility
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Using 4-bit quantization (BitsAndBytesConfig).")
                elif load_in_4bit:
                    logger.warning("CUDA not available, cannot load in 4-bit. Loading in default precision.")

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto", # Let accelerate handle device placement
                    torch_dtype=torch.bfloat16 if quantization_config else None, # Match compute dtype
                    trust_remote_code=True, # Required for some models
                )
                logger.info(f"Local model '{model_name}' loaded successfully.")
                cls._models_tokenizers[model_name] = (tokenizer, model)
            except Exception as e:
                logger.exception(f"Failed to load local model {model_name}")
                raise
        return cls._models_tokenizers[model_name]

def generate_response(
    model_name: str,
    prompt_text: str, # Changed name for clarity
    temperature: float = 1.0,
    max_new_tokens: int = 1500, # Renamed from max_tokens
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    load_in_4bit: bool = True,
    **gen_kwargs, # Capture other generation kwargs
):
    """Generates response using the loaded local model."""
    try:
        tokenizer, model = LocalLLMCache.get_model_tokenizer(model_name, load_in_4bit)

        # Encode the prompt
        inputs = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True)
        input_ids = inputs["input_ids"].to(model.device) # Move inputs to model's device
        attention_mask = inputs["attention_mask"].to(model.device)

        # Prepare generation arguments, filtering out None values
        generation_config = select_values(notnone, {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": tokenizer.eos_token_id, # Important for open-ended generation
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": temperature > 0.0, # Sample only if temperature is positive
            **gen_kwargs, # Include any other passed arguments
        })
        
        logger.info(f"Generating response with config: {generation_config}")

        # Generate
        with torch.no_grad(): # Inference mode
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )

        # Decode, skipping special tokens and the prompt
        # Need to handle potential differences in output structure/indexing
        # This assumes output contains the prompt; we slice it off.
        output_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        return output_text.strip()
    except Exception as e:
        logger.exception("Error generating response with local model")
        # Depending on desired behavior, could return None, empty string, or re-raise
        return f"Error during generation: {e}"


def query(
    system_message: str | None,
    user_message: str | None,
    model: str, # This is the model_name for HF
    temperature: float | None = None,
    max_tokens: int | None = None, # Corresponds to max_new_tokens
    func_spec: FunctionSpec | None = None, # Local models generally don't support this well yet
    convert_system_to_user: bool = False, # Less relevant if using chat templates
    hf_local: dict | None = None, # Specific config for HF local
    **kwargs, # Capture any remaining kwargs like top_p, top_k, etc.
) -> tuple[OutputType, float, int, int, dict]:
    """Query a local Hugging Face model."""
    if func_spec is not None:
         logger.warning("Function calling (func_spec) is generally not supported by local HF models via this backend. Ignoring.")

    # Process hf_local config or defaults
    hf_config = hf_local or {}
    use_chat_template = hf_config.get('use_chat_template', True) # Default to using chat template
    load_in_4bit = hf_config.get('load_in_4bit', True) # Default to 4-bit if possible

    # Prepare generation parameters
    gen_params = {
        "temperature": temperature if temperature is not None else 1.0,
        "max_new_tokens": max_tokens if max_tokens is not None else 1500,
        "top_p": kwargs.get('top_p', hf_config.get('top_p')),
        "top_k": kwargs.get('top_k', hf_config.get('top_k')),
        "repetition_penalty": kwargs.get('repetition_penalty', hf_config.get('repetition_penalty')),
        "load_in_4bit": load_in_4bit,
    }

    # Construct the prompt text
    prompt_input_text: str
    if use_chat_template:
        # Prepare messages for chat template
        messages = []
        if system_message:
             # If converting or if model prefers user role for system prompt
             if convert_system_to_user or hf_config.get('system_as_user', False):
                 messages.append({"role": "user", "content": system_message})
             else:
                 messages.append({"role": "system", "content": system_message})
        if user_message:
            messages.append({"role": "user", "content": user_message})

        # Load only the tokenizer to apply the template
        try:
            tokenizer, _ = LocalLLMCache.get_model_tokenizer(model, load_in_4bit=False) # Don't need full model yet
            prompt_input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            logger.info("Applied chat template.")
        except Exception as e:
            logger.warning(f"Failed to apply chat template for {model}: {e}. Falling back to simple concatenation.")
            # Fallback to simple concatenation
            prompt_parts = [msg for msg in [system_message, user_message] if msg]
            prompt_input_text = "\n\n".join(prompt_parts)
    else:
        # Simple concatenation if chat template is disabled
        prompt_parts = [msg for msg in [system_message, user_message] if msg]
        prompt_input_text = "\n\n".join(prompt_parts)

    logger.info(f"Final prompt text:\n{prompt_input_text[:500]}...") # Log start of prompt

    # Generate response
    t0 = time.time()
    response_text = generate_response(
        model_name=model,
        prompt_text=prompt_input_text,
        **gen_params
    )
    req_time = time.time() - t0

    # For local models, token counts are often not easily available or accurate without extra steps
    # Return 0 for now, or implement token counting if crucial
    in_tokens = 0
    out_tokens = 0
    info = {"model_name": model, "provider": "hf_local"}

    logger.info(f"Local response: {response_text[:500]}...")
    return response_text, req_time, in_tokens, out_tokens, info

# END OF MODIFIED FILE aide-ds/aide/backend/backend_local.py





# # backend_local.py
# import logging
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch

# logger = logging.getLogger("aide")

# class LocalLLMHandler:
#     _instance = None  # Singleton instance

#     def __new__(cls, model_name: str):
#         if cls._instance is None or cls._instance.model_name != model_name:
#             cls._instance = super(LocalLLMHandler, cls).__new__(cls)
#             cls._instance.model_name = model_name
#             cls._instance.tokenizer, cls._instance.model = cls._instance._load_model()
#         return cls._instance
    
#     def _load_model(self):
#         try:
#             logger.info(f"Loading local model: {self.model_name}")
#             tokenizer = AutoTokenizer.from_pretrained(self.model_name)

#             # Use BitsAndBytesConfig for quantization
#             quantization_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.float16,  # Adjust dtype if needed
#             )

#             model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name,
#                 quantization_config=quantization_config,
#                 device_map="auto"
#             )

#             logger.info(f"Quantized (4-bit) Local model {self.model_name} loaded successfully.")
#             return tokenizer, model

#         except Exception as e:
#             logger.error(f"Failed to load local model {self.model_name}: {e}")
#             raise
#     def generate_response(self, prompt: str, temperature: float = 1.0, max_tokens: int = 200):
#         try:
#             input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
#             output = self.model.generate(
#                 input_ids,
#                 max_length=max_tokens,
#                 temperature=temperature,
#                 do_sample=True,
#             )
#             return self.tokenizer.decode(output[0], skip_special_tokens=True)
#         except Exception as e:
#             logger.error(f"Error generating response: {e}")
#             return None

# def query(
#     system_message: str | None,
#     user_message: str | None,
#     model: str,
#     temperature: float | None = None,
#     max_tokens: int | None = None,
#     func_spec: None = None,
#     convert_system_to_user: bool = False,
#     **model_kwargs,
# ):
#     try:
#         if user_message is None:
#             prompt = system_message or ""
#         elif system_message is None:
#             prompt = user_message
#         else:
#             prompt = f"{system_message}\n\n{user_message}"

#         llm_handler = LocalLLMHandler(model)
#         response = llm_handler.generate_response(
#             prompt,
#             temperature=temperature or 1.0,  # Default temperature
#             max_tokens=max_tokens or 1500,  # Default max_tokens
#         )
#         return response, 0.0, 0, 0, {} # Add dummy values for other return values.

#     except Exception as e:
#         logger.error(f"Local query failed: {e}")
#         return None, 0.0, 0, 0, {}
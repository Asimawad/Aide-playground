# backend_local.py
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

logger = logging.getLogger("aide")

class LocalLLMHandler:
    _instance = None  # Singleton instance

    def __new__(cls, model_name: str):
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = super(LocalLLMHandler, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.tokenizer, cls._instance.model = cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        try:
            logger.info(f"Loading local model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Use BitsAndBytesConfig for quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # Adjust dtype if needed
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )

            logger.info(f"Quantized (4-bit) Local model {self.model_name} loaded successfully.")
            return tokenizer, model

        except Exception as e:
            logger.error(f"Failed to load local model {self.model_name}: {e}")
            raise
    def generate_response(self, prompt: str, temperature: float = 1.0, max_tokens: int = 200):
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(
                input_ids,
                max_length=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None

def query(
    system_message: str | None,
    user_message: str | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
):
    try:
        if user_message is None:
            prompt = system_message or ""
        elif system_message is None:
            prompt = user_message
        else:
            prompt = f"{system_message}\n\n{user_message}"

        llm_handler = LocalLLMHandler(model)
        response = llm_handler.generate_response(
            prompt,
            temperature=temperature or 1.0,  # Default temperature
            max_tokens=max_tokens or 1500,  # Default max_tokens
        )
        return response, 0.0, 0, 0, {} # Add dummy values for other return values.

    except Exception as e:
        logger.error(f"Local query failed: {e}")
        return None, 0.0, 0, 0, {}
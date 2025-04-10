# aide-ds/aide/backend/backend_vllm.py
"""Backend for vLLM OpenAI-compatible API."""

import json
import logging
import time
import os
from funcy import notnone, once, select_values
import openai
from omegaconf import OmegaConf # To read config if needed

from aide.backend.utils import (FunctionSpec, OutputType, opt_messages_to_list, backoff_create)

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore
_vllm_config: dict = {} # To store VLLM specific config

# Define potential exceptions (adjust if vLLM raises different ones)
VLLM_API_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIError,
    openai.InternalServerError
)

@once
def _setup_vllm_client():
    """Sets up the OpenAI client to point to the vLLM server."""
    global _client, _vllm_config
    try:
        # Attempt to load config if not already loaded (may need adjustment based on how config is passed)
        if not _vllm_config:
             # This assumes config is accessible globally or passed differently.
             # A cleaner way might be to pass cfg during the first query.
             # For now, we rely on environment variables or a hardcoded default.
             _vllm_config['base_url'] = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
             _vllm_config['api_key'] = os.getenv("VLLM_API_KEY", "EMPTY") # vLLM often uses 'EMPTY' or no key

        logger.info(f"Setting up vLLM client with base_url: {_vllm_config['base_url']}")
        _client = openai.OpenAI(
            base_url=_vllm_config['base_url'],
            api_key=_vllm_config['api_key'],
            max_retries=0 # Rely on backoff decorator
        )
    except Exception as e:
        logger.error(f"Failed to setup vLLM client: {e}", exc_info=True)
        raise

def set_vllm_config(cfg: OmegaConf):
    """Allows setting vLLM config explicitly, e.g., from the main run."""
    global _vllm_config
    if cfg.get('vllm'):
         _vllm_config['base_url'] = cfg.vllm.get('base_url', "http://localhost:8000/v1")
         _vllm_config['api_key'] = cfg.vllm.get('api_key', "EMPTY")
    else: # Fallback to environment or defaults if 'vllm' section is missing
         _vllm_config['base_url'] = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
         _vllm_config['api_key'] = os.getenv("VLLM_API_KEY", "EMPTY")


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query a model served via vLLM using the OpenAI-compatible endpoint.
    """
    _setup_vllm_client() # Ensure client is initialized

    # Filter out None kwargs and prepare messages
    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)

    # Handle function calling specification if provided
    tools_arg = None
    tool_choice_arg = None
    if func_spec is not None:
        # Check if the served model likely supports tools (heuristic)
        model_name = filtered_kwargs.get("model", "")
        if "instruct" in model_name.lower() or "tool" in model_name.lower(): # Basic check
            tools_arg = [{"type": "function", "function": func_spec.to_dict()}]
            # Forcing function call might depend on exact vLLM/model support.
            # Standard OpenAI forcing: {"type": "function", "function": {"name": func_spec.name}}
            # Some vLLM setups might need just "auto" or the specific name string. Test this.
            tool_choice_arg = {"type": "function", "function": {"name": func_spec.name}}
            logger.info("vLLM: Function spec provided, attempting tool call.")
        else:
            logger.warning("vLLM: Function spec provided, but model name doesn't suggest tool support. Skipping tool call.")
            func_spec = None # Disable func_spec if unsure

    # Add tools/tool_choice to kwargs if applicable
    if tools_arg:
        filtered_kwargs["tools"] = tools_arg
    if tool_choice_arg:
        filtered_kwargs["tool_choice"] = tool_choice_arg

    # Perform the API call with backoff
    t0 = time.time()
    try:
        completion = backoff_create(
            _client.chat.completions.create,
            VLLM_API_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except Exception as e:
         logger.error(f"vLLM query failed: {e}", exc_info=True)
         # Re-raise or return an error state if desired
         raise # Re-raise for now

    req_time = time.time() - t0

    # Process the response
    choice = completion.choices[0]
    message = choice.message

    output: OutputType
    if func_spec and message.tool_calls:
        # Process function call
        tool_call = message.tool_calls[0]
        if tool_call.function.name == func_spec.name:
            try:
                output = json.loads(tool_call.function.arguments)
                logger.info(f"vLLM: Successfully parsed tool call arguments for '{func_spec.name}'.")
            except json.JSONDecodeError as e:
                logger.error(f"vLLM: Error decoding JSON arguments for tool '{func_spec.name}': {tool_call.function.arguments}", exc_info=True)
                # Fallback to raw text or raise error? Let's return raw args for debugging.
                output = tool_call.function.arguments # Or raise e
        else:
            logger.error(f"vLLM: Tool call name mismatch. Expected '{func_spec.name}', got '{tool_call.function.name}'.")
            output = message.content or "" # Fallback to text content
    else:
        # Process regular text response
        output = message.content or ""

    # Extract token counts and other info (may vary with vLLM versions)
    input_tokens = completion.usage.prompt_tokens if completion.usage else 0
    output_tokens = completion.usage.completion_tokens if completion.usage else 0

    info = {
        "model": completion.model,
        "finish_reason": choice.finish_reason,
        # Add other relevant info if available from vLLM completion object
    }

    return output, req_time, input_tokens, output_tokens, info


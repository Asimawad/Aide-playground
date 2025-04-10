# START OF MODIFIED FILE aide-ds/aide/backend/__init__.py
import logging
from . import backend_anthropic, backend_local, backend_openai, backend_openrouter, backend_gdm, backend_deepseek, backend_vllm # Added backend_vllm
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

logger = logging.getLogger("aide")


def determine_provider(model: str) -> str:
    if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("o3-"):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("deepseek"): # Assuming direct API or a non-vLLM/non-local setup
        return "deepseek"
    elif model.startswith("gemini-"):
        return "gdm"
    elif model.startswith("vllm/"): # New condition for vLLM
        return "vllm"
    # all other models are handle by local (Hugging Face Transformers)
    else:
        return "local"


provider_to_query_func = {
    "openai": backend_openai.query,
    "anthropic": backend_anthropic.query,
    "gdm": backend_gdm.query,
    "openrouter": backend_openrouter.query,
    "deepseek": backend_deepseek.query,
    "vllm": backend_vllm.query, # Added vLLM mapping
    "local": backend_local.query,
}


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    # local_use : bool = False, # This parameter seems unused, consider removing
    reasoning_effort: str | None = None,
    # **model_kwargs, # Changed to pass all extra kwargs
    **kwargs, # Use kwargs to capture all extra arguments for flexibility
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message.
        user_message (PromptType | None): Uncompiled user message.
        model (str): string identifier for the model to use (e.g., "gpt-4-turbo", "vllm/deepseek-coder", "codellama/CodeLlama-7b-Instruct-hf").
        temperature (float | None, optional): Temperature to sample at.
        max_tokens (int | None, optional): Maximum number of tokens to generate.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call.
        convert_system_to_user (bool): Flag to convert system message role to user.
        reasoning_effort (str | None): Specific parameter for some models (like o3-mini).
        **kwargs: Additional keyword arguments passed directly to the backend query function (e.g., `hf_local`, `vllm` configs, `top_p`).

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    provider = determine_provider(model)
    logger.info(f"Determined provider: {provider} for model: {model}")

    model_query_kwargs = {
        "model": model, # Pass the specific model identifier
        "convert_system_to_user": convert_system_to_user,
        **kwargs # Pass through all other kwargs
    }

    # Handle standard parameters, allowing backend-specific ones to override if needed
    if temperature is not None:
        model_query_kwargs.setdefault("temperature", temperature)
    if max_tokens is not None:
        # Use provider-specific naming conventions if necessary, otherwise default
        if provider == "openai" and model.startswith("o3-"):
            model_query_kwargs.setdefault("max_completion_tokens", max_tokens)
        else:
            model_query_kwargs.setdefault("max_tokens", max_tokens)
            
    if provider == "openai" and model.startswith("o3-") and reasoning_effort:
         model_query_kwargs.setdefault("reasoning_effort", reasoning_effort)

    # Compile prompts
    compiled_system_message = compile_prompt_to_md(system_message) if system_message else None
    compiled_user_message = compile_prompt_to_md(user_message) if user_message else None

    # Log query details
    logger.info("---Querying model---", extra={"verbose": True})
    if compiled_system_message:
        logger.info(f"System: {compiled_system_message}", extra={"verbose": True})
    if compiled_user_message:
        logger.info(f"User: {compiled_user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"Function spec: {func_spec.to_dict()}", extra={"verbose": True})
    logger.info(f"Query Kwargs: {model_query_kwargs}", extra={"verbose": True})


    # Get and call the appropriate backend function
    query_func = provider_to_query_func.get(provider)
    if query_func is None:
        raise ValueError(f"Unsupported provider for model: {model}")

    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compiled_system_message,
        user_message=compiled_user_message,
        func_spec=func_spec,
        **model_query_kwargs,
    )

    # Log response details
    logger.info(f"Response Raw: {output}", extra={"verbose": True})
    logger.info(f"Time: {req_time:.2f}s, InTokens: {in_tok_count}, OutTokens: {out_tok_count}", extra={"verbose": True})
    logger.info(f"Info: {info}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})

    return output

# END OF MODIFIED FILE aide-ds/aide/backend/__init__.py
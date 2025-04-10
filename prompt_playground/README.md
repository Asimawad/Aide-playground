# START OF NEW FILE aide-ds/prompt_playground/README.md
# Prompt Playground for AIDE Models

This directory provides a simplified environment to test prompts with local Hugging Face models, execute the generated code, and view the results. It's designed for rapid iteration on prompt engineering, especially for smaller open-source models.

## Setup

1.  **Create Environment:** It's recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements-playground.txt
    # Make sure you have torch installed appropriately for your hardware (CPU/CUDA)
    # e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
3.  **Ensure AIDE is Accessible:** If running this script from the `prompt_playground` directory, the main `aide` package needs to be discoverable. The easiest way is often to install AIDE in editable mode from the project root (`aide-ds/`):
    ```bash
    pip install -e .
    ```
    Alternatively, adjust the `sys.path` logic within `prompt_playground.py` if needed.

## Usage

```bash
python prompt_playground/prompt_playground.py \
    --model_id "deepseek-ai/DeepSeek-Coder-V2-Instruct" \
    --prompt_file "prompt_playground/sample_prompt.txt" \
    --output_dir . \
    --num_responses 1 \
    --temperature 0.5 \
    --max_new_tokens 2024 \
    [--top_p <P>] \
    [--top_k <K>] \
    [--repetition_penalty <R>] \
    [--execution_timeout <S>] \
    [--no_4bit]


Arguments:
--model_id: (Required) The Hugging Face identifier of the model to load (e.g., "deepseek-ai/deepseek-coder-7b-instruct-v1.5").
--prompt_file: (Required) Path to the text file containing the prompt. See sample_prompt.txt for format (use ---USER--- to separate system/initial user message from the main user request).
--output_dir: (Required) Directory where generated code, logs, and submission files will be saved for each response.
--num_responses (Optional): Number of independent responses to generate for the prompt (default: 1).
--temperature (Optional): Sampling temperature (default: 0.2).
--max_new_tokens (Optional): Max tokens to generate (default: 2048).
--top_p (Optional): Nucleus sampling p (default: None).
--top_k (Optional): Top-k sampling k (default: None).
--repetition_penalty (Optional): Repetition penalty (default: None).
--execution_timeout (Optional): Timeout in seconds for code execution (default: 60).
--no_4bit: (Optional) Disable 4-bit quantization (load in full precision). Requires significant VRAM/RAM.


Prompt File Format:
The script expects a simple text file for the prompt:
Lines before ---USER--- are treated as the system prompt (or initial user turns in a chat).
Lines after ---USER--- are treated as the main user request/instruction.
If ---USER--- is not found, the entire file content is used as the user prompt.
Output:
For each response generated (response_0, response_1, etc.), the script will create subdirectories within the specified output_dir. Each subdirectory will contain:
raw_response.txt: The full text output from the LLM.
extracted_code.py: The Python code extracted from the response.
execution_log.json: A JSON file containing the ExecutionResult (stdout, stderr, errors, execution time).
submission/submission.csv: If the code successfully generated this file.
working/: Any temporary files created by the executed code.
The console will print a summary for each response indicating success or failure of execution and the paths to the saved files.
Note on Parallelism:
This script generates multiple responses sequentially. For true parallel inference leveraging GPU power efficiently, consider using a dedicated inference server like vLLM and adapting the script (or the main AIDE vLLM backend) to send multiple requests concurrently.
END OF NEW FILE aide-ds/prompt_playground/README.md
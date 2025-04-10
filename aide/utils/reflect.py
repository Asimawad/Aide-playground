# START OF MODIFIED FILE aide-ds/aide/utils/self_reflection.py

from typing import Any, Callable
import re
import logging

logger = logging.getLogger("aide")

# Define necessary type hints for the functions being passed
QueryFuncType = Callable[..., str] # Simplified type hint for query
WrapCodeFuncType = Callable[[str, str], str] # Added lang argument
ExtractCodeFuncType = Callable[[str], str] # Simplified type hint for extract_code

# --- Few-Shot Example Definition ---
# Example Input Code (with a common error)
EXAMPLE_INPUT_CODE = """
import pandas as pd
import numpy as np

# Simulate loading data
try:
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
    sample_submission = pd.read_csv('./input/sample_submission.csv')
except FileNotFoundError:
    # Fallback for testing if files aren't present
    train_df = pd.DataFrame({'feature1': np.random.rand(100), 'target': np.random.randint(0, 2, 100), 'id': range(100)})
    test_df = pd.DataFrame({'feature1': np.random.rand(50), 'id': range(100, 150)})
    sample_submission = pd.DataFrame({'id': range(100, 150), 'target': np.zeros(50)})

# Simulate predictions (replace with actual model logic)
# Ensuring predictions match the length of the test set
predictions = np.random.rand(len(test_df))

# Create submission DataFrame using test_df's id column
submission_df = pd.DataFrame({'id': test_df['id'], 'target': predictions})

# Save submission - INCORRECT PATH & FILENAME
submission_df.to_csv('my_submission.csv', index=False) # <<< ERROR HERE

print("Submission file created.")
print("Validation Metric: 0.85") # Example metric print
"""

# Expected Output for Stage 1 (Critique) based on EXAMPLE_INPUT_CODE
EXAMPLE_STAGE1_OUTPUT_CRITIQUE = """The main mistake is saving the submission file with an incorrect name and in the wrong directory, it must be './submission/submission.csv'.
1. Line 22: Change the filename from `'my_submission.csv'` to `'submission.csv'`.
2. Line 22: Change the file path to include the target directory, making it `'./submission/submission.csv'`.
"""

# Expected Output for Stage 2 (Code Edit) based on EXAMPLE_INPUT_CODE and EXAMPLE_STAGE1_OUTPUT_CRITIQUE
EXAMPLE_STAGE2_OUTPUT_CODE = """# Applying edits based on review.
```python
import pandas as pd
import numpy as np

# Simulate loading data
try:
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
    sample_submission = pd.read_csv('./input/sample_submission.csv')
except FileNotFoundError:
    # Fallback for testing if files aren't present
    train_df = pd.DataFrame({'feature1': np.random.rand(100), 'target': np.random.randint(0, 2, 100), 'id': range(100)})
    test_df = pd.DataFrame({'feature1': np.random.rand(50), 'id': range(100, 150)})
    sample_submission = pd.DataFrame({'id': range(100, 150), 'target': np.zeros(50)})

# Simulate predictions (replace with actual model logic)
# Ensuring predictions match the length of the test set
predictions = np.random.rand(len(test_df))

# Create submission DataFrame using test_df's id column
submission_df = pd.DataFrame({'id': test_df['id'], 'target': predictions})

# Save submission - CORRECTED PATH & FILENAME
submission_df.to_csv('./submission/submission.csv', index=False) # <<< FIXED HERE

print("Submission file created.")
print("Validation Metric: 0.85") # Example metric print

```"""
# --- End Few-Shot Example Definition ---

def perform_two_step_reflection_with_fewshot(
    code: str,
    task_desc: str,
    model_name: str,
    temperature: float,
    convert_system_to_user: bool,
    query_func: QueryFuncType,
    wrap_code_func: WrapCodeFuncType,
    extract_code_func: ExtractCodeFuncType,
    **kwargs # Capture other args like hf_local, vllm
) -> tuple[str, str]:
    """
    Performs a two-step self-reflection with a few-shot example included in prompts.

    1. Critiques the code and proposes minimal text-based edits (guided by an example).
    2. Applies only those edits to the original code (guided by an example).

    Args:
        code: The code string to reflect upon.
        task_desc: The description of the task for context.
        model_name: Name of the language model to use.
        temperature: Temperature setting for the language model.
        convert_system_to_user: Flag for handling system messages.
        query_func: The function used to query the language model.
        wrap_code_func: Function to wrap code for prompts.
        extract_code_func: Function to extract code from LLM responses.
        **kwargs: Additional arguments to pass to the query function.

    Returns:
        Tuple: (reflection_plan, revised_code)
               - reflection_plan: Text describing the critique and planned edits.
               - revised_code: The minimally revised code, or original if no changes/errors.
    """
    # --- Stage 1: Critique and Edit Proposal (with Few-Shot Example) ---
    critique_prompt = {
        "Role": "You are a meticulous Python code reviewer focused on finding small, critical errors.",
        "Task": (
            "1. Carefully review the 'Code to Review' section below.\n"
            "2. Identify 1 to 4 specific, small mistakes. Focus on: typos, incorrect variable names, simple logic errors (e.g., off-by-one), incorrect file paths (especially the required `./submission/submission.csv`), or missing imports.\n"
            "3. Write a concise summary sentence explaining the main error(s).\n"
            "4. Provide step-by-step, text-only instructions (numbered list) to fix ONLY those specific mistakes.\n"
            "5. If you find NO specific errors that need changing, respond ONLY with the exact sentence: No specific errors found requiring changes.\n"
            "6. STRICTLY follow the format shown in the 'EXAMPLE'."
        ),
        "EXAMPLE": (
            "### EXAMPLE START ###\n"
            "Input Code:\n"
            f"{wrap_code_func(EXAMPLE_INPUT_CODE, lang='python')}\n\n" # Use wrap_code_func
            "Expected Output:\n"
            f"{EXAMPLE_STAGE1_OUTPUT_CRITIQUE}\n"
            "### EXAMPLE END ###"
        ),
        "Rules": (
            "RULE 1: **ABSOLUTELY NO PYTHON CODE in your response.** Only text instructions.\n"
            "RULE 2: Only fix small, clear mistakes. Do NOT suggest improvements, refactoring, or new features.\n"
            "RULE 3: Ensure file paths match requirements (e.g., `./submission/submission.csv`).\n"
            "RULE 4: Follow the 'Output Format' EXACTLY, like the example."
        ),
        "Output Format": (
            "If mistakes are found:\n"
            "- Start with one sentence explaining the main mistake(s).\n"
            "- Then, write a NUMBERED list of fix instructions.\n"
            "- Each number is ONE simple step (e.g., '1. Line 17: Change `my_submission.csv` to `./submission/submission.csv`.').\n"
            "\n"
            "If no mistakes are found:\n"
            "- Write ONLY this sentence: No specific errors found requiring changes."
        ),
        "Code to Review": wrap_code_func(code, lang='python'), # Use the passed function
        "Context": task_desc,
    }

    logger.debug(f"Critique Prompt (Stage 1):\n{critique_prompt}")

    plan_raw = query_func(
        system_message=critique_prompt,
        user_message=None, # User message can be empty if prompt is detailed
        model=model_name,
        temperature=temperature, # Consider lower temp (e.g., 0.1) for critique
        convert_system_to_user=convert_system_to_user,
        **kwargs # Pass extra args
    )
    reflection_plan = re.sub(r"<think>.*?</think>", "", plan_raw, flags=re.DOTALL).strip()
    logger.debug(f"Critique Response (Stage 1 Raw):\n{plan_raw}")
    logger.info(f"Critique Plan (Stage 1 Cleaned):\n{reflection_plan}")


    if not reflection_plan or reflection_plan.strip() == "No specific errors found requiring changes.":
        logger.info("Reflection Step 1: No changes suggested.")
        return reflection_plan or "No specific errors found requiring changes.", code

    # --- Stage 2: Focused Code Edit (with Few-Shot Example) ---
    coder_prompt = {
        "Role": "You are a precise code editor. You only apply the given text instructions.",
        "Task": (
            "1. Take the 'Original Code'.\n"
            "2. Apply *ONLY* the changes described in the 'Edit Instructions' (which are text, not code).\n"
            "3. Output the *entire, modified* code within a single Python code block.\n"
            "4. Follow the 'Output Format' EXACTLY."
        ),
        "EXAMPLE": (
            "### EXAMPLE START ###\n"
            "Original Code:\n"
            f"{wrap_code_func(EXAMPLE_INPUT_CODE, lang='python')}\n\n"
            "Edit Instructions:\n"
            f"{EXAMPLE_STAGE1_OUTPUT_CRITIQUE}\n\n"
            "Expected Output:\n"
            f"{EXAMPLE_STAGE2_OUTPUT_CODE}\n"
            "### EXAMPLE END ###"
        ),
        "Rules": (
            "RULE 1: Apply ONLY the numbered steps from 'Edit Instructions'.\n"
            "RULE 2: **DO NOT change any other part of the code.** No reformatting, no adding comments (except the first required one), no restructuring.\n"
            "RULE 3: Ignore any Python code examples *within* the 'Edit Instructions'; only follow the text steps.\n"
            "RULE 4: Your entire output MUST follow the 'Output Format'."
        ),
        "Output Format": (
            "# Applying edits based on review.\n" # MUST be the very first line
            "```python\n"
            "[The FULL original code, with ONLY the requested edits applied]\n"
            "```\n"
            "**IMPORTANT: NO TEXT before the first '#' comment. NO TEXT after the final '```'.**"
        ),
        "Original Code": wrap_code_func(code, lang='python'),
        "Edit Instructions": reflection_plan,
    }

    logger.debug(f"Coder Prompt (Stage 2):\n{coder_prompt}")

    revised_code_response = query_func(
        system_message=coder_prompt,
        user_message=None,
        model=model_name,
        temperature=0.0, # Use temp 0 for deterministic editing
        convert_system_to_user=convert_system_to_user,
        **kwargs # Pass extra args
    )

    logger.debug(f"Coder Response (Stage 2 Raw):\n{revised_code_response}")
    revised_code = extract_code_func(revised_code_response)

    # Post-processing: Remove the mandatory comment if it exists
    if revised_code and revised_code.startswith("# Applying edits based on review.\n"):
        revised_code = revised_code.split('\n', 1)[1]
        # Also remove potential leading/trailing whitespace again after splitting
        revised_code = revised_code.strip() if revised_code else ""


    if not revised_code:
        logger.warning("Reflection Step 2: Code extraction failed. Returning original code.")
        return reflection_plan, code
    else:
        logger.info("Reflection Step 2: Successfully generated revised code.")
        return reflection_plan, revised_code

# END OF MODIFIED FILE aide-ds/aide/utils/self_reflection.py
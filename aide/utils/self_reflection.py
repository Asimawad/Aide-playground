# Inside ./utils/self_reflection.py
import shutil
import logging
import random
import time
from typing import Any, Callable, cast
import re
import humanize
from ..backend import FunctionSpec, compile_prompt_to_md, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from .config import Config
from .metric import MetricValue, WorstMetricValue
from .response import extract_code, extract_text_up_to_code, wrap_code

# Assuming query, wrap_code, extract_code are accessible or passed in.
# Define necessary type hints for the functions being passed
QueryFuncType = Callable[..., str] # Simplified type hint for query
WrapCodeFuncType = Callable[[str], str] # Simplified type hint for wrap_code
ExtractCodeFuncType = Callable[[str], str] # Simplified type hint for extract_code

def perform_two_step_reflection(
    code: str,
    task_desc: str,
    model_name: str,
    temperature: float,
    convert_system_to_user: bool,
    query_func: QueryFuncType,
    wrap_code_func: WrapCodeFuncType,
    extract_code_func: ExtractCodeFuncType
) -> tuple[str, str]:
    """
    Performs a two-step self-reflection on the provided code.

    1. Critiques the code and proposes minimal text-based edits.
    2. Applies only those edits to the original code.

    Args:
        code: The code string to reflect upon.
        task_desc: The description of the task for context.
        model_name: Name of the language model to use.
        temperature: Temperature setting for the language model.
        convert_system_to_user: Flag for handling system messages.
        query_func: The function used to query the language model.
        wrap_code_func: Function to wrap code for prompts.
        extract_code_func: Function to extract code from LLM responses.

    Returns:
        Tuple: (reflection_plan, revised_code)
               - reflection_plan: Text describing the critique and planned edits.
               - revised_code: The minimally revised code, or original if no changes.
    """
    # Stage 1: Critique and Edit Proposal (Prompt from your Agent.reflect)
    critique_prompt = {
        # --- ROLE ---
        "Role": "You are a simple code checker.",
        # --- TASK ---
        "Task": (
            "1. Read the 'Code to Review' below.\n"
            "2. Find 1 to 4 small mistakes (like typos, wrong variable names, simple logic errors, bad file paths).\n"
            "3. Write step-by-step text instructions to fix ONLY those small mistakes.\n"
            "4. If you find NO mistakes, just write the single sentence: No specific errors found requiring changes."
        ),
        # --- CODE TO REVIEW ---
        "Code to Review": wrap_code_func(code), # Use the passed function
        # --- CONTEXT ---
        "Context": task_desc, # Use the passed task_desc
        # --- RULES ---
        "Rules": (
            "RULE 1: **DO NOT WRITE ANY PYTHON CODE IN YOUR RESPONSE.**\n"
            "RULE 2: Do not suggest big changes or new ideas.\n"
            "RULE 3: Only suggest fixes for small mistakes in the code shown.\n"
            "RULE 4: Follow the 'Output Format' EXACTLY."
        ),
        # --- OUTPUT FORMAT ---
        "Output Format": (
            "If mistakes are found:\n"
            "- Start with one sentence explaining the main mistake(s).\n"
            "- Then, write a NUMBERED list of fix instructions.\n"
            "- Each number is ONE simple step.\n"
            "- Example Step: '1. Line 15: Change `pred_y` to `predictions`.'\n"
            "- Example Step: '2. Line 30: Change file path to `./submission/output.csv`.'\n"
            "\n"
            "If no mistakes are found:\n"
            "- Write only this sentence: No specific errors found requiring changes."
            "\n"
            "**FINAL CHECK: Did you include any Python code? If yes, REMOVE IT.**"
        )
    }
    plan_raw = query_func( # Use the passed function
        system_message=critique_prompt,
        user_message=None,
        model=model_name, # Use the passed argument
        temperature=temperature, # Use the passed argument
        convert_system_to_user=convert_system_to_user, # Use the passed argument
    )
    # Clean the plan (e.g., remove think tags if your query function adds them)
    reflection_plan = re.sub(r"<think>.*?</think>", "", plan_raw, flags=re.DOTALL).strip()

    # Check if critique suggested no changes
    if reflection_plan.strip() == "No specific errors found requiring changes.":
        # Return the original code as no changes are needed
        return reflection_plan, code

    # Stage 2: Focused Code Edit (Prompt from your Agent.reflect)
    coder_prompt = {
        # --- ROLE ---
        "Role": "You are a precise code editor.",
        # --- TASK ---
        "Task": (
            "1. Read the 'Original Code'.\n"
            "2. Read the 'Edit Instructions'. These are text instructions, NOT code.\n"
            "3. Apply ONLY the changes from 'Edit Instructions' to the 'Original Code'.\n"
            "4. Output the result EXACTLY as shown in 'Output Format'."
        ),
        # --- ORIGINAL CODE ---
        "Original Code": wrap_code_func(code), # Use the passed function
        # --- EDIT INSTRUCTIONS ---
        "Edit Instructions": reflection_plan, # Use the cleaned plan from Stage 1
        # --- RULES ---
        "Rules": (
            "RULE 1: Apply ONLY the steps from 'Edit Instructions'.\n"
            "RULE 2: **Do NOT change any other part of the code.**\n"
            "RULE 3: Do not reformat or restructure code.\n"
            "RULE 4: If 'Edit Instructions' accidentally contains any code examples, IGNORE THEM. Follow only the numbered text steps.\n"
            "RULE 5: Your entire output MUST follow the 'Output Format'."
        ),
        # --- OUTPUT FORMAT ---
        "Output Format": (
            "Line 1: Start IMMEDIATELY with a comment '# Applying edits based on review.'\n"
            "Next Line: Start the Python code block immediately.\n"
            "```python\n"
            "[The FULL original code, with ONLY the requested edits applied]\n"
            "```\n"
            "**IMPORTANT: NO TEXT before the first '#' comment. NO TEXT after the final '```'.**"
        )
    }

    revised_code_response = query_func( # Use the passed function
        system_message=coder_prompt,
        user_message=None,
        model=model_name, # Use the passed argument
        temperature=temperature, # Use the passed argument
        convert_system_to_user=convert_system_to_user, # Use the passed argument
    )
    revised_code = extract_code_func(revised_code_response) # Use the passed function

    # Return original code if extraction fails or is empty, otherwise revised
    return reflection_plan, revised_code if revised_code else code


# Example Input Code (with a common error)
EXAMPLE_INPUT_CODE = """
import pandas as pd
import numpy as np

# Simulate loading data
train_df = pd.DataFrame({'feature1': np.random.rand(100), 'target': np.random.randint(0, 2, 100)})
test_df = pd.DataFrame({'feature1': np.random.rand(50)})
sample_submission = pd.DataFrame({'id': range(50), 'target': np.zeros(50)})

# Simulate predictions
predictions = np.random.rand(len(test_df))

# Create submission DataFrame
submission_df = pd.DataFrame({'id': sample_submission['id'], 'target': predictions})

# Save submission - INCORRECT PATH
submission_df.to_csv('my_submission.csv', index=False) # <<< ERROR HERE

print("Submission file created.")
"""

# Expected Output for Stage 1 (Critique) based on EXAMPLE_INPUT_CODE
EXAMPLE_STAGE1_OUTPUT_CRITIQUE = """The primary mistake is saving the submission file with an incorrect name and potentially in the wrong directory relative to the expected './submission/' folder.
1. Line 17: Change the filename from `'my_submission.csv'` to `'submission.csv'`.
2. Line 17: Change the file path to include the target directory, making it `'./submission/submission.csv'`.
""" # NOTE: Combined steps for clarity, could be separate. Adjust based on model performance.

# Expected Output for Stage 2 (Code Edit) based on EXAMPLE_INPUT_CODE and EXAMPLE_STAGE1_OUTPUT_CRITIQUE
EXAMPLE_STAGE2_OUTPUT_CODE = """# Applying edits based on review.
```python
import pandas as pd
import numpy as np

# Simulate loading data
train_df = pd.DataFrame({'feature1': np.random.rand(100), 'target': np.random.randint(0, 2, 100)})
test_df = pd.DataFrame({'feature1': np.random.rand(50)})
sample_submission = pd.DataFrame({'id': range(50), 'target': np.zeros(50)})

# Simulate predictions
predictions = np.random.rand(len(test_df))

# Create submission DataFrame
submission_df = pd.DataFrame({'id': sample_submission['id'], 'target': predictions})

# Save submission - CORRECTED PATH
submission_df.to_csv('./submission/submission.csv', index=False) # <<< FIXED HERE

print("Submission file created.")
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
    extract_code_func: ExtractCodeFuncType
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

    Returns:
        Tuple: (reflection_plan, revised_code)
               - reflection_plan: Text describing the critique and planned edits.
               - revised_code: The minimally revised code, or original if no changes/errors.
    """
    # --- Stage 1: Critique and Edit Proposal (with Few-Shot Example) ---
    critique_prompt = {
        # --- ROLE ---
        "Role": "You are a simple code checker.",
        # --- TASK ---
        "Task": (
            "1. Read the 'Code to Review' below.\n"
            "2. Find 1 to 4 small mistakes (like typos, wrong variable names, simple logic errors, bad file paths).\n"
            "3. Write step-by-step text instructions to fix ONLY those small mistakes.\n"
            "4. If you find NO mistakes, just write the single sentence: No specific errors found requiring changes.\n"
            "5. Follow the format shown in the 'EXAMPLE' precisely."
        ),
         # --- EXAMPLE START ---
        "EXAMPLE": (
            "### EXAMPLE START ###\n"
            "Input Code:\n"
            f"{wrap_code_func(EXAMPLE_INPUT_CODE)}\n\n" # Use wrap_code_func for consistency
            "Expected Output:\n"
            f"{EXAMPLE_STAGE1_OUTPUT_CRITIQUE}\n"
            "### EXAMPLE END ###"
        ),
        # --- RULES ---
        "Rules": (
            "RULE 1: **DO NOT WRITE ANY PYTHON CODE IN YOUR RESPONSE.**\n"
            "RULE 2: Do not suggest big changes or new ideas.\n"
            "RULE 3: Only suggest fixes for small mistakes in the code shown.\n"
            "RULE 4: Follow the 'Output Format' EXACTLY, like the example."
        ),
        # --- OUTPUT FORMAT ---
        "Output Format": (
            "If mistakes are found:\n"
            "- Start with one sentence explaining the main mistake(s).\n"
            "- Then, write a NUMBERED list of fix instructions.\n"
            "- Each number is ONE simple step.\n"
            "- Example Step: '1. Line 15: Change `pred_y` to `predictions`.'\n"
            "- Example Step: '2. Line 30: Change file path to `./submission/output.csv`.'\n"
            "\n"
            "If no mistakes are found:\n"
            "- Write only this sentence: No specific errors found requiring changes."
            "\n"
            "**FINAL CHECK: Did you include any Python code? If yes, REMOVE IT.**"
        ),
        # --- ACTUAL CODE TO REVIEW ---
        "Code to Review": wrap_code_func(code), # Use the passed function
        # --- CONTEXT ---
        "Context": task_desc, # Use the passed task_desc
    }

    # Log the prompt being sent (optional, but helpful for debugging)
    # logging.debug(f"Critique Prompt (Stage 1):\n{critique_prompt}")

    plan_raw = query_func( # Use the passed function
        system_message=critique_prompt,
        user_message=None, # Assuming system message contains everything
        model=model_name,
        temperature=temperature,
        convert_system_to_user=convert_system_to_user,
    )
    # Clean the plan (e.g., remove think tags if your query function adds them)
    reflection_plan = re.sub(r"<think>.*?</think>", "", plan_raw, flags=re.DOTALL).strip()

    # Check if critique suggested no changes
    if reflection_plan.strip() == "No specific errors found requiring changes." or not reflection_plan:
        # logging.info("Reflection Step 1: No changes suggested.")
        return reflection_plan, code # Return original code

    # --- Stage 2: Focused Code Edit (with Few-Shot Example) ---
    coder_prompt = {
        # --- ROLE ---
        "Role": "You are a precise code editor.",
        # --- TASK ---
        "Task": (
            "1. Read the 'Original Code'.\n"
            "2. Read the 'Edit Instructions'. These are text instructions, NOT code.\n"
            "3. Apply ONLY the changes from 'Edit Instructions' to the 'Original Code'.\n"
            "4. Output the result EXACTLY as shown in the 'EXAMPLE' and 'Output Format'."
        ),
         # --- EXAMPLE START ---
        "EXAMPLE": (
            "### EXAMPLE START ###\n"
            "Original Code:\n"
            f"{wrap_code_func(EXAMPLE_INPUT_CODE)}\n\n" # Use wrap_code_func
            "Edit Instructions:\n"
            f"{EXAMPLE_STAGE1_OUTPUT_CRITIQUE}\n\n"
            "Expected Output:\n"
            f"{EXAMPLE_STAGE2_OUTPUT_CODE}\n" # Note: This already includes the comment and code block
            "### EXAMPLE END ###"
        ),
        # --- RULES ---
        "Rules": (
            "RULE 1: Apply ONLY the steps from 'Edit Instructions'.\n"
            "RULE 2: **Do NOT change any other part of the code.**\n"
            "RULE 3: Do not reformat or restructure code.\n"
            "RULE 4: If 'Edit Instructions' accidentally contains any code examples, IGNORE THEM. Follow only the numbered text steps.\n"
            "RULE 5: Your entire output MUST follow the 'Output Format', like the example."
        ),
        # --- OUTPUT FORMAT ---
        "Output Format": (
            "Line 1: Start IMMEDIATELY with a comment '# Applying edits based on review.'\n"
            "Next Line: Start the Python code block immediately.\n"
            "```python\n"
            "[The FULL original code, with ONLY the requested edits applied]\n"
            "```\n"
            "**IMPORTANT: NO TEXT before the first '#' comment. NO TEXT after the final '```'.**"
        ),
         # --- ACTUAL INPUTS FOR THIS TASK ---
        "Original Code": wrap_code_func(code), # Use the passed function
        "Edit Instructions": reflection_plan, # Use the cleaned plan from Stage 1
    }

    # Log the prompt being sent (optional)
    # logging.debug(f"Coder Prompt (Stage 2):\n{coder_prompt}")

    revised_code_response = query_func( # Use the passed function
        system_message=coder_prompt,
        user_message=None, # Assuming system message contains everything
        model=model_name,
        temperature=temperature, # Consider using low temp (e.g., 0.0) for precise editing
        convert_system_to_user=convert_system_to_user,
    )
    revised_code = extract_code_func(revised_code_response) # Use the passed function

    # Return original code if extraction fails or is empty, otherwise revised
    if not revised_code:
        # logging.warning("Reflection Step 2: Code extraction failed. Returning original code.")
        return reflection_plan, code
    else:
        # logging.info("Reflection Step 2: Successfully generated revised code.")
        return reflection_plan, revised_code


























# # self reflection logic
# def _reflect(
#     code: str,
#     task_desc: str,
#     model_name: str,
#     temperature: float,
#     convert_system_to_user: bool,
#     query_func: QueryFuncType,
#     wrap_code_func: WrapCodeFuncType,
#     extract_code_func: ExtractCodeFuncType
#     reflection_steps:int):
    
#     introduction = (
#         "You are a Kaggle grandmaster attending a competition. "
#         "Your task is to review your code and check for potential bugs. For example, check of the code produces a csv submission in the correct path ./submission/submission.csv file, "
#         "so based on the information below, you should revise it in order to fix potential bugs. "
#         "Your response should be an improved implementation outline in natural language,"
#         " followed by a single markdown code block in which you keep the parts of the code that do not need modifications, and implements the bugfix/solution where needed."
#         "this markdown code should be a copy of the previous code, and only modify the parts that need to be changed. in order not to induce bugs that were not in the original code."
#     )
#     prompt: Any = {
#         "Introduction": introduction,
#         "Original Task description": self.task_desc,
#         "Previous (not revised) implementation": wrap_code(code),
#         "Response format": (
#             "Your response should be a brief outline/sketch of (original solution + modification) in natural language (3-5 sentences), "
#             "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
#             "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block containing the revised code. "
#         ),
#         "Instructions": {},
#     }

#     reflection_plan, reflection_code = self.plan_and_code_query(prompt)
#     return reflection_plan, reflection_code
# def _reflect(self, code):
#     """Generate a natural language reflection plan + code in the same LLM call and split them apart."""
#     introduction = (
#         "You are a Kaggle grandmaster attending a competition. Your task is to review your code to check for potential bugs, "
#         "look at the methods and imports for possible helucinations"
#         "with particular attention to ensuring that the test dataset is not operated on using non-existent fields (e.g., some feature does not exist in the test set) and that the submission.csv file is saved correctly in the './submission/' directory. "
#         "Identify and explain any mistakes, but leave all code lines that are correct completely unchanged. "
#         "In your response, first provide a brief explanation (3–5 sentences) of the identified issues and how you fixed them, "
#         "and then output a single markdown code block that is an exact copy of the original code except for the minimal modifications necessary to correct the errors. "
#         "Do not modify any parts of the code that do not require changes."
#     )
#     prompt = {
#         "Introduction": introduction,
#         "Original Task description": self.task_desc,
#         "Previous (not revised) implementation": wrap_code(code),
#         "Response format": (
#             "Your response should be a brief outline/sketch (3–5 sentences) explaining the modifications, "
#             "followed by a single markdown code block (wrapped in ```) that is an exact copy of the original code with only the necessary changes to fix the identified bugs."
#         ),
#         "Instructions": {}
#     }
#     reflection_plan, reflection_code = self.plan_and_code_query(prompt)
#     return reflection_plan, reflection_code
# def _multi_step_reflect(self, code, reflection_steps=3):
#     prompts = ["Reflection Iteration 1:You are a Kaggle grandmaster attending a competition. Your task is to review the provided Python code intended to solve the competition task described below. In this first round, focus on identifying obvious bugs and potential hallucinations—such as referencing non-existent fields in the test dataset or errors in saving the submission.csv file in the './submission/' directory. Use the task details from the competition description (self.task_desc) to guide your review. In your response, first provide a brief explanation (3–5 sentences) of the identified issues and the modifications you made, and then output a single markdown code block that is an exact copy of the original code with only the minimal modifications necessary to fix these bugs. Do not change any code that is already correct.",
#     "Reflection Iteration 2:Building on the code from Iteration 1, review the revised code for any residual issues. In this round, concentrate on ensuring that variable usage, data transformations, and all method imports adhere to the requirements stated in the competition description (self.task_desc). Also, check that no unintended modifications were introduced in the previous iteration. Provide a concise explanation (3–5 sentences) detailing the additional refinements you made, and then output a single markdown code block containing only the minimal modifications made relative to the Iteration 1 version.",
#     ",Reflection Iteration 3:Using the code from Iteration 2 as your starting point, perform a final, comprehensive review for complete correctness and robustness. Confirm that the code fully meets the specifications from the competition description (self.task_desc), including proper data handling, consistency in method usage, and correct creation of the submission file. Provide a brief summary (3–5 sentences) of any final corrections or enhancements, and output a single markdown code block that includes only the minimal changes from the Iteration 2 code necessary to address any remaining issues."]
#     current_code = code
#     for i in range(reflection_steps):
#         prompt = {
#             "Introduction": prompts[i],
#             "Original Task description": self.task_desc,
#             "Previous (not revised) implementation": wrap_code(current_code),
#             "Response format": (
#                 "Your response should be a brief outline/sketch (3–5 sentences) explaining the modifications, "
#                 "followed by a single markdown code block (wrapped in ```) that is an exact copy of the original code with only the necessary changes to fix the identified bugs."
#             ),
#             "Instructions": {}
#         }
#         reflection_plan, current_code = self.plan_and_code_query(prompt)
#         # Check if the code is empty or contains only whitespace
#         if not current_code.strip():
#             logger.info("Reflection code is empty or contains only whitespace.")
#             break
#         # Check if the code is the same as the previous iteration
#         if current_code == code:
#             logger.info("Reflection code is the same as the previous iteration.")
#             break
#     return reflection_plan, current_code

# def I_reflect(self, code: str) -> tuple[str, str]:
#     """        Two-step self-reflection:
#     1. The model critiques the code and produces a list of minimal edits.
#     2. The model applies those edits only, and outputs a minimally modified version of the original code.
    
#     Returns:
#         Tuple: (reflection_plan, revised_code)"""
#     # Stage 1: Critique and Edit Proposal
#     # Prompt for Step 1: Critique Code (Designed for ~7B LLM)
#     critique_prompt = {
#         # --- ROLE ---
#         "Role": "You are a simple code checker.",

#         # --- TASK ---
#         "Task": (
#             "1. Read the 'Code to Review' below.\n"
#             "2. Find 1 to 4 small mistakes (like typos, wrong variable names, simple logic errors, bad file paths).\n"
#             "3. Write step-by-step text instructions to fix ONLY those small mistakes.\n"
#             "4. If you find NO mistakes, just write the single sentence: No specific errors found requiring changes."
#         ),

#         # --- CODE TO REVIEW ---
#         "Code to Review": wrap_code(code),
#         # Assuming self.task_desc provides simple context, add it if necessary:
#         "Context": self.task_desc,

#         # --- RULES ---
#         "Rules": (
#             "RULE 1: **DO NOT WRITE ANY PYTHON CODE IN YOUR RESPONSE.**\n"  # Primary, simple, strong constraint
#             "RULE 2: Do not suggest big changes or new ideas.\n"
#             "RULE 3: Only suggest fixes for small mistakes in the code shown.\n"
#             "RULE 4: Follow the 'Output Format' EXACTLY."
#         ),

#         # --- OUTPUT FORMAT ---
#         "Output Format": (
#             "If mistakes are found:\n"
#             "- Start with one sentence explaining the main mistake(s).\n"
#             "- Then, write a NUMBERED list of fix instructions.\n"
#             "- Each number is ONE simple step.\n"
#             "- Example Step: '1. Line 15: Change `pred_y` to `predictions`.'\n" # Example-like instruction
#             "- Example Step: '2. Line 30: Change file path to `./submission/output.csv`.'\n" # Example-like instruction
#             "\n"
#             "If no mistakes are found:\n"
#             "- Write only this sentence: No specific errors found requiring changes."
#             "\n"
#             "**FINAL CHECK: Did you include any Python code? If yes, REMOVE IT.**" # Final reinforcement
#         )
#     }
            
#     plan = query(
#         system_message=critique_prompt,
#         user_message=None,
#         model=self.acfg.code.model,
#         temperature=self.acfg.code.temp,
#         convert_system_to_user=self.acfg.convert_system_to_user,
#     )
#     reflection_plan = re.sub(r"<think>.*?</think>", "", plan, flags=re.DOTALL)
#     # reflection_plan = re.findall(r"<think>(.*?)</think>", plan, flags=re.DOTALL)

#     # Stage 2: Focused Code Edit
#     # Prompt for Step 2: Apply Edits (Designed for ~7B LLM)
#     coder_prompt = {
#         # --- ROLE ---
#         "Role": "You are a precise code editor.",

#         # --- TASK ---
#         "Task": (
#             "1. Read the 'Original Code'.\n"
#             "2. Read the 'Edit Instructions'. These are text instructions, NOT code.\n"
#             "3. Apply ONLY the changes from 'Edit Instructions' to the 'Original Code'.\n"
#             "4. Output the result EXACTLY as shown in 'Output Format'."
#         ),

#         # --- ORIGINAL CODE ---
#         "Original Code": wrap_code(code),

#         # --- EDIT INSTRUCTIONS ---
#         "Edit Instructions": reflection_plan, # Text output from Step 1

#         # --- RULES ---
#         "Rules": (
#             "RULE 1: Apply ONLY the steps from 'Edit Instructions'.\n"
#             "RULE 2: **Do NOT change any other part of the code.**\n"
#             "RULE 3: Do not reformat or restructure code.\n"
#             "RULE 4: If 'Edit Instructions' accidentally contains any code examples, IGNORE THEM. Follow only the numbered text steps.\n" # Crucial for robustness
#             "RULE 5: Your entire output MUST follow the 'Output Format'."
#         ),

#         # --- OUTPUT FORMAT ---
#         "Output Format": (
#             "Line 1: Start IMMEDIATELY with a comment '# Applying edits based on review.'\n" # Simple, predefined comment
#             # Optional: Add 1-2 more comment lines if needed, keep simple.
#             # "Line 2: # Fixing identified issues in logic/paths.\n"
#             "Next Line: Start the Python code block immediately.\n"
#             "```python\n"
#             "[The FULL original code, with ONLY the requested edits applied]\n"
#             "```\n"
#             "**IMPORTANT: NO TEXT before the first '#' comment. NO TEXT after the final '```'.**" # Reinforce structure
#         )
#     }
            
#     revised_code_response = query(
#         system_message=coder_prompt,
#         user_message=None,
#         model=self.acfg.code.model,
#         temperature=self.acfg.code.temp,
#         convert_system_to_user=self.acfg.convert_system_to_user,
#     )
#     revised_code = extract_code(revised_code_response)
    
#     return reflection_plan, revised_code

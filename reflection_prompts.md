# my first reflection prompt:
reflection_prompt = {
    "Introduction": "You are a meticulous Kaggle grandmaster reviewing your own Python code for a house price prediction competition.",
    "Task description": self.task_desc,
    "Code to review": wrap_code(result_node.code),
    "Instructions": {
        "Review guideline": [
            "Focus only on ensuring all necessary libraries (e.g., pandas, numpy, scikit-learn) are imported at the top.",
            "Do not modify correct parts of the code—only add or fix missing/invalid imports.",
            "Check for invalid or made-up import names (e.g., non-existent packages or misspelled names)."
        ],
        "Response format": (
            "List specific import issues found (e.g., 'Missing import pandas', 'Invalid import xyz') in 1-2 sentences. "
            "Then, provide the *updated code block* with only the necessary changes to fix imports, "
            "preserving all other code exactly as is. Use comments to indicate changes (e.g., '# Added import pandas'). "
            "If no issues, say 'No import issues found' and repeat the original code unchanged."
        )
    }
}

## Issues: The biggest issue is that this process is a bit chaotic, it is trying to tackle each and every single issue in the code at once, this results in the following issues, :
1. the solutions for the problems are basically not that thourough "you cannot be accurate when trying to solve many things at once.
2. the second major problem is that the model is forgetting stuff! for example, if there was a line in the original code that was correct, 


**for example** the line that saves the csv file , the model is ok by not focusing on it in the reflection process as it is already correct, but when creating the final code after reflection, the model forgets, as if it is writing all over again and not just correcting the missing things and leaving the other things untouched.

# second reflection prompt
**Goal**: Make the prompt crystal clear, ensuring DeepSeek 7B modifies only what’s necessary, preserves correct parts, and focuses on one or two issues at a time in each reflection.

reflection_steps = 3
  # list of codes "evolution and enhancements of the code"
codes = []
current_code = result_node.code


 reflection_prompt = {
            "Introduction": "You are a Kaggle grandmaster reviewing your own Python code for a competition.",
            "Task description": self.task_desc,
            "Code to review": wrap_code(result_node.code),
            "Instructions": {
                "Review guideline": [
                    "Ensure all necessary libraries (e.g., pandas, numpy, scikit-learn) are imported at the top.",
                    "Verify input files are read from './input/' (e.g., './input/train.csv', './input/test.csv').",
                    "Confirm predictions are saved to './submission/submission.csv'—check the path and file creation.",
                    "Check for invalid or made-up methods/attributes (e.g., pandas or sklearn functions that don’t exist).",
                    "Fix any preprocessing issues, like handling missing values for both numerical and categorical columns."
                ],
                "Response format": (
                    "List specific issues found (e.g., 'Missing import pandas', 'Wrong path') in 2-3 sentences. "
                    "Then, provide the FULL corrected code—including all imports, data loading, preprocessing, "
                    "model training, validation, and submission saving—in a single code block. "
                    "If no issues, say 'No issues found' and repeat the original code unchanged."
                )
            }
        }
        
        reflection_plan, reflection_code = self.plan_and_code_query(reflection_prompt)
        if reflection_code:  # If DeepSeek provides a fix
            result_node.code = reflection_code
            logger.info(f"Node {result_node.id} self-reflected and updated code")

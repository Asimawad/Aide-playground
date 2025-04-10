# START OF NEW FILE aide-ds/finetuning/prepare_data.py
import json
import argparse
from pathlib import Path
import logging
import sys
from typing import List, Dict, Optional

# Add the parent directory of 'aide' to sys.path to allow importing aide modules
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root.parent))

try:
    from aide.journal import Journal, Node
    from aide.utils.serialize import load_json
except ImportError as e:
    print(f"Error importing AIDE modules: {e}")
    print("Please ensure the script is run from within the project structure or the 'aide-ds' directory is in the Python path.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_alpaca(instruction: str, input_str: Optional[str], output: str) -> Dict:
    """Formats data into Alpaca instruction-following format."""
    if input_str:
        text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n{output}"
    else:
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
    # You might want separate fields depending on the trainer library (e.g., SFTTrainer often takes 'text' or instruction/output directly)
    return {"text": text, "instruction": instruction, "input": input_str or "", "output": output}

def create_instruction_pairs(journal: Journal, task_desc: str) -> List[Dict]:
    """
    Extracts instruction-following pairs from the AIDE journal.
    Focuses on code generation steps (draft, improve, debug).
    """
    pairs = []
    nodes_by_id = {node.id: node for node in journal.nodes}

    for node in journal.nodes:
        # Skip nodes without code or plan (shouldn't happen often)
        if not node.code or not node.plan:
            continue

        instruction = ""
        input_str = ""
        output = node.code # The generated code is always the target output

        # --- Handle Draft Nodes ---
        if node.parent is None:
            instruction = f"Given the following machine learning task description, first write a brief plan and then the Python code to implement it.\n\nTask Description:\n{task_desc}"
            input_str = f"Plan:\n{node.plan}" # Include the agent's plan as input context
            if node.plan and node.code: # Only add if both plan and code exist
                 pairs.append(format_alpaca(instruction, input_str, output))

        # --- Handle Improve/Debug Nodes ---
        elif node.parent:
            parent_node = nodes_by_id.get(node.parent.id)
            if not parent_node:
                logger.warning(f"Parent node {node.parent.id} not found for node {node.id}. Skipping.")
                continue

            if parent_node.is_buggy: # Debugging Step
                instruction = f"The following Python code for a machine learning task has a bug, resulting in the error output shown. Briefly explain the fix and provide the corrected full Python code.\n\nTask Description:\n{task_desc}"
                input_str = (f"Buggy Code:\n```python\n{parent_node.code}\n```\n\n"
                             f"Execution Output/Error:\n```\n{parent_node.term_out}\n```\n\n" # Use raw term_out
                             f"Plan for Fix:\n{node.plan}")
                if node.plan and node.code:
                     pairs.append(format_alpaca(instruction, input_str, output))

            else: # Improvement Step
                instruction = f"Improve the following Python code for a machine learning task based on the provided plan. Output the complete, improved Python code.\n\nTask Description:\n{task_desc}"
                input_str = (f"Previous Code:\n```python\n{parent_node.code}\n```\n\n"
                             f"Improvement Plan:\n{node.plan}")
                if node.plan and node.code:
                     pairs.append(format_alpaca(instruction, input_str, output))

        # Could potentially add pairs for self-reflection critique/edit steps here
        # if node.reflection_plan and node.original_code: ...

    logger.info(f"Extracted {len(pairs)} instruction pairs from journal.")
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Prepare AIDE journal data for fine-tuning.")
    parser.add_argument("log_dir", type=str, help="Directory containing AIDE experiment log folders (each with a journal.json).")
    parser.add_argument("output_file", type=str, help="Path to save the output JSONL file.")
    parser.add_argument("--task_desc_file", type=str, default=None, help="Optional: Path to a single task description file to use for all journals.")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    global_task_desc = None
    if args.task_desc_file:
        try:
            with open(args.task_desc_file, 'r') as f:
                global_task_desc = f.read()
            logger.info(f"Using global task description from: {args.task_desc_file}")
        except FileNotFoundError:
            logger.error(f"Global task description file not found: {args.task_desc_file}")
            sys.exit(1)

    total_pairs = 0
    with open(output_file, 'w') as outfile:
        for exp_dir in log_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            journal_path = exp_dir / "journal.json"
            config_path = exp_dir / "config.yaml" # To get task desc if not global

            if journal_path.exists():
                logger.info(f"Processing journal: {journal_path}")
                try:
                    journal: Journal = load_json(journal_path, Journal)

                    task_desc = global_task_desc
                    if not task_desc:
                        # Try loading task desc from individual experiment config
                        if config_path.exists():
                            from omegaconf import OmegaConf # Lazy import
                            cfg = OmegaConf.load(config_path)
                            if cfg.get('desc_file') and Path(cfg.desc_file).exists():
                                with open(cfg.desc_file, 'r') as f:
                                    task_desc = f.read()
                            elif cfg.get('goal'):
                                task_desc = f"Goal: {cfg.goal}"
                                if cfg.get('eval'):
                                    task_desc += f"\nEvaluation: {cfg.eval}"
                            else:
                                logger.warning(f"Could not determine task description for {exp_dir}. Skipping.")
                                continue
                        else:
                            logger.warning(f"Config file not found for {exp_dir}, cannot get task description. Skipping.")
                            continue

                    instruction_pairs = create_instruction_pairs(journal, task_desc)
                    for pair in instruction_pairs:
                        outfile.write(json.dumps(pair) + '\n')
                    total_pairs += len(instruction_pairs)
                except Exception as e:
                    logger.error(f"Failed to process journal {journal_path}: {e}", exc_info=True)
            else:
                logger.warning(f"Journal file not found in {exp_dir}")

    logger.info(f"Finished processing. Total instruction pairs written: {total_pairs}")
    logger.info(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
# END OF NEW FILE aide-ds/finetuning/prepare_data.py
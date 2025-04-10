# START OF MODIFIED FILE aide-ds/aide/agent.py

import shutil
import logging
import random
import time
from typing import Any, Callable, cast
import numpy as np
import humanize
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code
# Import the few-shot reflection function
from .utils.self_reflection import perform_two_step_reflection_with_fewshot

logger = logging.getLogger("aide")


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a `submission.csv` file in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (1-2 sentences) describing "
                " the empirical findings (metric value). Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.", # Simplified
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully and printed a metric, report the value of the validation metric. Otherwise, leave it null.", # Simplified
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (e.g., MSE, RMSE), false if the metric should be maximized (e.g., accuracy, F1).", # Simplified
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        # Determine prompt style based on config
        self.use_simple_prompts = self.acfg.prompt_style == "simple"
        logger.info(f"Using prompt style: {'simple' if self.use_simple_prompts else 'default'}")

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        if greedy_node is None: # Should not happen if good_nodes is not empty, but for safety
             logger.warning("[search policy] No best node found despite having good nodes. Drafting.")
             return None
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    @property
    def _prompt_environment(self):
        # Keep environment prompt simple, less crucial for OS models
        pkgs = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "torch"]
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])
        env_prompt = {
            "Installed Packages": f"You can use common ML packages like: {pkg_str}. Most standard packages are installed."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = max(0, self.acfg.time_limit - tot_time_elapsed) # Ensure non-negative
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining)) if tot_time_remaining > 0 else self.cfg.exec.timeout

        impl_guideline = [
            # Time/Step info might confuse smaller models, make it simpler or remove
            # f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
            # f"<TOTAL_STEPS_REMAINING: {self.acfg.steps - self.current_step}>",
            "**Code Requirements:**",
            "- Implement the solution described.",
            "- Print the validation metric value (e.g., using `print(f'Validation Metric: {metric_value}')`).",
            "- **CRITICAL: Save test predictions to `./submission/submission.csv`.** This exact path and filename is required.",
            "- Write a single, self-contained Python script.",
            "- Ensure the script runs to completion without errors.",
            "- Be mindful of execution time (limit: ~{humanize.naturaldelta(exec_timeout)}).",
            "- Input data is in the `./input/` directory.",
            "- Use `./working/` for temporary files if needed.",
            "- **REMEMBER: Output `./submission/submission.csv` is mandatory!**",
        ]
        if self.acfg.expose_prediction:
            # Keep simple for OS models
            impl_guideline.append(
                "- Include a `predict()` function for reuse on new data."
            )

        if self.acfg.k_fold_validation > 1:
             impl_guideline.append(
                 f"- If appropriate for the task, use {self.acfg.k_fold_validation}-fold cross-validation for evaluation."
             )

        return {"Implementation Guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        # Make the format extremely explicit for OS models
        return {
            "Response Format": (
                "1. First, write a brief plan (3-5 sentences) outlining your solution approach.\n"
                "2. **Immediately after the plan**, start the Python code block like this:\n"
                "```python\n"
                "[YOUR PYTHON CODE HERE]\n"
                "```\n"
                "**IMPORTANT: There should be NO text after the final ```.**"
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for i in range(retries):
            try:
                 # Pass necessary config to the query function
                 completion_text = query(
                     system_message=prompt,
                     user_message=None,
                     model=self.acfg.code.model,
                     temperature=self.acfg.code.temp,
                     convert_system_to_user=self.acfg.convert_system_to_user,
                     # Pass relevant parts of config if needed by backends
                     hf_local=self.cfg.get('hf_local'),
                     vllm=self.cfg.get('vllm'),
                 )
                 code = extract_code(completion_text)
                 nl_text = extract_text_up_to_code(completion_text)

                 if code and nl_text:
                     logger.info("Plan and code extracted successfully.")
                     return nl_text, code
                 else:
                      logger.warning(f"Plan/Code extraction failed (Attempt {i+1}/{retries}). NL: {bool(nl_text)}, Code: {bool(code)}. Raw response: {completion_text[:200]}...")

            except Exception as e:
                 logger.error(f"Error during plan_and_code_query (Attempt {i+1}/{retries}): {e}", exc_info=True)
                 time.sleep(2) # Wait a bit before retrying after an error

        logger.error("Final plan + code extraction attempt failed. Returning empty plan and raw response as code.")
        # Return raw text as code if extraction fails completely
        return "", completion_text or ""


    def _draft(self) -> Node:
        if self.use_simple_prompts:
            introduction = (
                 "You are an expert Python programmer writing a machine learning solution. "
                 "Outline your plan briefly, then implement it in Python code."
            )
            solution_sketch_guideline = [
                 "Keep this first solution relatively simple.",
                 "Base your plan on the Task Description and Data Overview.",
                 "Describe your plan in 3-5 sentences.",
                 "Choose a reasonable evaluation metric if not specified.",
                 "Focus on model training and prediction generation.",
                 "Data is ready in `./input/`. No need to unzip.",
             ]
        else:
             # Original complex prompt
             introduction = (
                 "You are a Kaggle grandmaster attending a competition. "
                 "In order to win this competition, you need to come up with an excellent and creative plan "
                 "for a solution and then implement this solution in Python. We will now provide a description of the task."
             )
             if self.acfg.obfuscate:
                 introduction = (
                     "You are an expert machine learning engineer attempting a task. "
                     "In order to complete this task, you need to come up with an excellent and creative plan "
                     "for a solution and then implement this solution in Python. We will now provide a description of the task."
                 )
             solution_sketch_guideline = [
                 "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                 "Take the Memory section into consideration when proposing the design,"
                 " don't propose the same modelling solution but keep the evaluation the same.",
                 "The solution sketch should be 3-5 sentences.",
                 "Propose an evaluation metric that is reasonable for this task.",
                 "Don't suggest to do EDA.",
                 "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
             ]

        # Simplify Memory for OS models if needed (e.g., only last or best attempt)
        memory_summary = self.journal.generate_summary()
        if self.use_simple_prompts and len(self.journal.good_nodes) > 0:
             best_prev = self.journal.get_best_node()
             if best_prev:
                  memory_summary = f"Best Previous Attempt:\nDesign: {best_prev.plan}\nResults: {best_prev.analysis}\nValidation Metric: {best_prev.metric.value}\n"
             else:
                  memory_summary = "No successful previous attempts recorded."
        elif self.use_simple_prompts:
             memory_summary = "No previous attempts recorded yet."


        prompt: Any = {
            "Introduction": introduction,
            "Task Description": self.task_desc,
            # Use simplified memory for simple prompts
            "Memory / Previous Attempts": memory_summary,
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"]["Solution Plan Guideline"] = solution_sketch_guideline
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview and self.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        if not code: # If code extraction failed, log and create node with empty code
             logger.error("Drafting failed to produce valid code. Creating node with empty code.")
             code = "# Error: Failed to generate code."

        new_node = Node(plan=plan, code=code)
        logger.info(f"Drafted new node {new_node.id}")
        return new_node

    def _improve(self, parent_node: Node) -> Node:
        if self.use_simple_prompts:
             introduction = (
                 "You are an expert Python programmer improving a machine learning solution. "
                 "You are given the previous code and its results. "
                 "Briefly outline a SINGLE specific improvement, then rewrite the FULL Python code implementing ONLY that change."
             )
             improvement_sketch_guideline = [
                 "Focus on ONE small, specific improvement (e.g., change model, add feature engineering step, tune one hyperparameter).",
                 "Explain your proposed improvement in 2-4 sentences.",
                 "Refer to the 'Previous Solution Code' and 'Task Description'.",
                 "Consider the 'Memory / Previous Attempts' to avoid repeating failed ideas.",
             ]
        else:
            # Original complex prompt
            introduction = (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            )
            if self.acfg.obfuscate:
                introduction = (
                    "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                    "solution below and should improve it in order to further increase the (test time) performance. "
                    "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                    "then implement this improvement in Python based on the provided previous solution. "
                )
            improvement_sketch_guideline = [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ]
        # Simplify Memory for OS models if needed
        memory_summary = self.journal.generate_summary()
        if self.use_simple_prompts:
             best_prev = self.journal.get_best_node() # May not be the parent
             if best_prev and best_prev != parent_node:
                 memory_summary = (
                     f"Parent Attempt (Metric: {parent_node.metric.value}):\nPlan: {parent_node.plan}\nResults: {parent_node.analysis}\n\n"
                     f"Best Overall Attempt (Metric: {best_prev.metric.value}):\nPlan: {best_prev.plan}\nResults: {best_prev.analysis}"
                 )
             else:
                 memory_summary = f"Parent Attempt (Metric: {parent_node.metric.value}):\nPlan: {parent_node.plan}\nResults: {parent_node.analysis}\n"

        prompt: Any = {
            "Introduction": introduction,
            "Task Description": self.task_desc,
            # Use simplified memory for simple prompts
            "Memory / Previous Attempts": memory_summary,
            "Previous Solution Code": wrap_code(parent_node.code), # Always include parent code
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"]["Improvement Plan Guideline"] = improvement_sketch_guideline
        prompt["Instructions"] |= self._prompt_impl_guideline
        # Environment less critical for improvement? Optional.
        # prompt["Instructions"] |= self._prompt_environment

        plan, code = self.plan_and_code_query(prompt)
        if not code:
             logger.error(f"Improvement step for node {parent_node.id} failed to produce code.")
             code = "# Error: Failed to generate improved code."
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
        if self.use_simple_prompts:
             introduction = (
                 "You are an expert Python programmer debugging code. "
                 "The previous code failed or produced an error, shown in the 'Execution Output'. "
                 "Briefly explain the likely cause and the fix in your plan, then provide the corrected FULL Python code."
             )
             bugfix_sketch_guideline = [
                 "Identify the error from the 'Execution Output'.",
                 "Explain the likely cause and your fix in 2-4 sentences.",
                 "Refer to the 'Buggy Code'.",
             ]
        else:
            # Original complex prompt
            introduction = (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug and/or did not produce a submission.csv, "
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )
            if self.acfg.obfuscate:
                 introduction = (
                     "You are an expert machine learning engineer attempting a task. "
                     "Your previous solution had a bug and/or did not produce a submission.csv, "
                     "so based on the information below, you should revise it in order to fix this. "
                     "Your response should be an implementation outline in natural language,"
                     " followed by a single markdown code block which implements the bugfix/solution."
                 )
            bugfix_sketch_guideline = [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ]

        prompt: Any = {
            "Introduction": introduction,
            "Task Description": self.task_desc,
            "Buggy Code": wrap_code(parent_node.code),
            "Execution Output / Error": wrap_code(parent_node.term_out, lang=""), # Use raw term_out for debugging
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"]["Bugfix Plan Guideline"] = bugfix_sketch_guideline
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview and self.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        if not code:
             logger.error(f"Debugging step for node {parent_node.id} failed to produce code.")
             code = "# Error: Failed to generate debugged code."
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def reflect(self, code: str) -> tuple[str, str]:
        """
        Performs a two-step self-reflection using the few-shot utility function.
        """
        logger.info("Initiating two-step self-reflection (few-shot)...")
        try:
            # Use the few-shot version
            reflection_plan, revised_code = perform_two_step_reflection_with_fewshot(
                code=code,
                task_desc=self.task_desc,
                model_name=self.acfg.code.model, # Use coding model for reflection edits
                temperature=self.acfg.code.temp, # Use coding temp
                convert_system_to_user=self.acfg.convert_system_to_user,
                query_func=query,
                wrap_code_func=wrap_code,
                extract_code_func=extract_code,
                 # Pass relevant parts of config if needed by backends
                hf_local=self.cfg.get('hf_local'),
                vllm=self.cfg.get('vllm'),
            )

            # Check plan content to log appropriately
            if reflection_plan.strip() == "No specific errors found requiring changes.":
                logger.info("Self-reflection found no specific errors requiring changes.")
            elif revised_code and revised_code != code:
                 logger.info("Self-reflection resulted in code changes.")
                 # logger.debug(f"Reflection Plan:\n{reflection_plan}") # Optionally log plan
            else:
                 # This case means a plan was generated but the code didn't change or extraction failed
                 logger.warning("Self-reflection generated a plan but no code changes were applied (or extraction failed).")
                 # logger.debug(f"Reflection Plan:\n{reflection_plan}") # Log the plan for debugging

            # Return original code if revised_code is empty or same as original
            return reflection_plan, revised_code if revised_code and revised_code != code else code

        except Exception as e:
            logger.error(f"Error during self-reflection: {e}", exc_info=True)
            # Fallback to original code in case of error
            return "Reflection error occurred.", code


    def update_data_preview(
        self,
    ):
        try:
             self.data_preview = data_preview.generate(self.cfg.workspace_dir / "input") # Generate from input subdir
             logger.info("Data preview updated.")
        except Exception as e:
             logger.error(f"Failed to generate data preview: {e}", exc_info=True)
             self.data_preview = "Error: Could not generate data preview."


    def step(self, exec_callback: ExecCallbackType):
        # Clear the submission dir
        submission_dir = self.cfg.workspace_dir / "submission"
        shutil.rmtree(submission_dir, ignore_errors=True)
        submission_dir.mkdir(exist_ok=True)

        if self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.info(f"Agent step {self.current_step+1}/{self.acfg.steps}. Parent node: {parent_node.id if parent_node else 'None'}")

        draft_flag = False
        if parent_node is None:
            draft_flag = True
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        # Perform self-reflection, especially on drafts or if enabled for all steps
        # For OS models, reflecting on the initial draft is often beneficial.
        # if draft_flag: # Reflect only on drafts
        # Reflect on every generated code attempt before execution
        reflection_plan, reflected_code = self.reflect(code=result_node.code)
        if reflected_code != result_node.code:
            logger.info(f"Node {result_node.id} code updated after self-reflection.")
            # Optionally store the plan: result_node.reflection_plan = reflection_plan
            result_node.code = reflected_code
        else:
             logger.info(f"Node {result_node.id} code unchanged after self-reflection.")


        # Proceed with execution
        logger.info(f"Agent executing code for node {result_node.id}")
        exec_result = exec_callback(result_node.code, True) # Always reset session for simplicity now
        result_node = self.parse_exec_result(node=result_node, exec_result=exec_result)

        # Check if submission.csv was actually created
        submission_file = self.cfg.workspace_dir / "submission" / "submission.csv"
        has_submission = submission_file.exists() and submission_file.stat().st_size > 0

        if not result_node.is_buggy and not has_submission:
            result_node.is_buggy = True
            result_node.metric = WorstMetricValue()
            result_node.analysis = (result_node.analysis or "") + "\nError: submission.csv not found or empty in ./submission/ directory."
            logger.warning(f"Node {result_node.id} marked as buggy: submission.csv missing/empty.")
        elif not result_node.is_buggy and has_submission:
             result_node.analysis = (result_node.analysis or "") + "\nInfo: submission.csv successfully created."
        elif result_node.is_buggy and has_submission:
             # Buggy for other reasons, but submission exists
             result_node.analysis = (result_node.analysis or "") + "\nWarning: submission.csv created, but node marked as buggy due to other errors/metric issues."

        self.journal.append(result_node)

        # Cache best solution
        best_node = self.journal.get_best_node()
        if best_node is not None and best_node.id == result_node.id:
            logger.info(f"Node {result_node.id} is the new best node (Metric: {result_node.metric}). Caching solution.")
            best_solution_dir = self.cfg.workspace_dir / "best_solution"
            best_solution_dir.mkdir(exist_ok=True, parents=True)
            best_submission_dir = self.cfg.workspace_dir / "best_submission" # Keep separate maybe?
            best_submission_dir.mkdir(exist_ok=True, parents=True)

            # Copy submission.csv if it exists
            if submission_file.exists():
                 try:
                      shutil.copy2(submission_file, best_submission_dir / "submission.csv")
                 except Exception as e:
                      logger.error(f"Failed to copy best submission.csv: {e}")

            # Save best solution code and node id
            try:
                with open(best_solution_dir / "solution.py", "w") as f:
                    f.write(result_node.code)
                with open(best_solution_dir / "node_id.txt", "w") as f:
                    f.write(str(result_node.id))
            except Exception as e:
                 logger.error(f"Failed to write best solution files: {e}")
        elif best_node is not None:
            logger.info(f"Node {result_node.id} (Metric: {result_node.metric}) is not better than best node {best_node.id} (Metric: {best_node.metric}).")
        else:
             logger.info(f"Node {result_node.id} processed. No valid best node yet.")

        self.current_step += 1


    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        """Parses execution result using LLM feedback."""
        logger.info(f"Agent parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        # Simplify intro for OS models if needed
        if self.use_simple_prompts:
             introduction = (
                 "You are reviewing the output of a Python script for an ML task. "
                 "Determine if it failed (bug), if it created `./submission/submission.csv`, and report the validation metric value."
             )
        else:
            # Original complex prompt
            introduction = (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
            if self.acfg.obfuscate:
                 introduction = (
                     "You are an expert machine learning engineer attempting a task. "
                     "You have written code to solve this task and now need to evaluate the output of the code execution. "
                     "You should determine if there were any bugs as well as report the empirical findings."
                 )

        prompt = {
            "Introduction": introduction,
            "Task Description": self.task_desc, # Provide task context
            "Code Executed": wrap_code(node.code),
            "Execution Output Log": wrap_code(node.term_out, lang=""), # Use raw term_out
        }

        try:
             # Use feedback model specified in config
             response = cast(
                 dict,
                 query(
                     system_message=prompt,
                     user_message="Evaluate the execution using the 'submit_review' function.", # Explicit instruction
                     func_spec=review_func_spec,
                     model=self.acfg.feedback.model,
                     temperature=self.acfg.feedback.temp,
                     convert_system_to_user=self.acfg.convert_system_to_user,
                     # Pass relevant parts of config if needed by backends
                     hf_local=self.cfg.get('hf_local'),
                     vllm=self.cfg.get('vllm'),
                 ),
             )
        except Exception as e:
             logger.error(f"Error during feedback query for node {node.id}: {e}", exc_info=True)
             # Handle error: Mark as buggy, provide default analysis
             node.analysis = f"Error during feedback analysis: {e}"
             node.is_buggy = True
             node.metric = WorstMetricValue()
             return node

        # Validate response structure (basic check)
        if not all(k in response for k in ["is_bug", "has_csv_submission", "summary", "metric", "lower_is_better"]):
             logger.error(f"Feedback response for node {node.id} has missing keys: {response}")
             node.analysis = "Error: Feedback LLM response format incorrect."
             node.is_buggy = True
             node.metric = WorstMetricValue()
             return node


        # Check if metric is a valid number
        metric_value = response.get("metric")
        is_metric_valid = isinstance(metric_value, (int, float)) and not np.isnan(metric_value) and not np.isinf(metric_value)

        # Determine buggy status
        # Buggy if: LLM says it is, code execution raised an exception, metric is invalid,
        # OR LLM says no submission.csv was produced (we double-check this later in step())
        node.is_buggy = (
            response.get("is_bug", True) # Default to buggy if key missing
            or node.exc_type is not None
            or not is_metric_valid
            or not response.get("has_csv_submission", False) # If LLM claims no submission, mark buggy for now
        )
        node.analysis = response.get("summary", "No analysis provided.")

        if node.is_buggy:
            logger.info(f"Parsed results: Node {node.id} is buggy.")
            if node.exc_type:
                 logger.info(f"Reason: Exception during execution ({node.exc_type}).")
            elif not is_metric_valid:
                 logger.info(f"Reason: Invalid metric value ({metric_value}).")
            elif not response.get("has_csv_submission", False):
                 logger.info("Reason: Feedback LLM reported no submission.csv (will be verified).")
            else:
                 logger.info("Reason: Feedback LLM reported is_bug=true.")
            node.metric = WorstMetricValue()
        else:
            # Ensure lower_is_better is boolean, default to None if missing/invalid
            lower_is_better = response.get("lower_is_better")
            maximize_metric = not lower_is_better if isinstance(lower_is_better, bool) else None
            
            node.metric = MetricValue(
                value=float(metric_value), # Already validated as float/int
                maximize=maximize_metric
            )
            logger.info(f"Parsed results: Node {node.id} is not buggy. Metric: {node.metric}")
            if maximize_metric is None:
                 logger.warning(f"Optimization direction (lower_is_better) for metric on node {node.id} is unknown or invalid.")

        return node

# END OF MODIFIED FILE aide-ds/aide/agent.py
import shutil
import logging
import random
import time
from typing import Any, Callable, cast

import humanize
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

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
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
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
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = self.acfg.time_limit - tot_time_elapsed
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        impl_guideline = [
            f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
            f"<TOTAL_STEPS_REMAINING: {self.acfg.steps - self.current_step}>",
            "The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE `./submission/` DIRECTORY.**",            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the "./submission" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            "You can use the `./working/` directory to store temporary files (e.g., models, intermediate data), but the final `submission.csv` MUST be in `./submission/`.",
            "REMEMBER THE `./submission/submission.csv` FILE!!!!! The correct directory is critical for evaluation.",
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

        
                
    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""

        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
            )
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
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
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview
        # print("____________________________________________________\n")
        # print(f"the currently used Prompt: {compile_prompt_to_md(prompt)}")
        # print("\n____________________________________________________\n")
        
        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code)
        logger.info(f"Drafted new node {new_node.id}")
        return new_node

    def _improve(self, parent_node: Node) -> Node:
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
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
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
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id}")
        return new_node
    
    # self reflection logic
    # def _reflect(self,code, multi_step= False, reflection_steps=3):
    #     """Generate a natural language reflection plan + code in the same LLM call and split them apart."""
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
    def _reflect(self, code):
        """Generate a natural language reflection plan + code in the same LLM call and split them apart."""
        introduction = (
            "You are a Kaggle grandmaster attending a competition. Your task is to review your code to check for potential bugs, "
            "look at the methods and imports for possible helucinations"
            "with particular attention to ensuring that the test dataset is not operated on using non-existent fields (e.g., some feature does not exist in the test set) and that the submission.csv file is saved correctly in the './submission/' directory. "
            "Identify and explain any mistakes, but leave all code lines that are correct completely unchanged. "
            "In your response, first provide a brief explanation (3–5 sentences) of the identified issues and how you fixed them, "
            "and then output a single markdown code block that is an exact copy of the original code except for the minimal modifications necessary to correct the errors. "
            "Do not modify any parts of the code that do not require changes."
        )
        prompt = {
            "Introduction": introduction,
            "Original Task description": self.task_desc,
            "Previous (not revised) implementation": wrap_code(code),
            "Response format": (
                "Your response should be a brief outline/sketch (3–5 sentences) explaining the modifications, "
                "followed by a single markdown code block (wrapped in ```) that is an exact copy of the original code with only the necessary changes to fix the identified bugs."
            ),
            "Instructions": {}
        }
        reflection_plan, reflection_code = self.plan_and_code_query(prompt)
        return reflection_plan, reflection_code
    def _multi_step_reflect(self, code, reflection_steps=3):
        prompts = ["Reflection Iteration 1:You are a Kaggle grandmaster attending a competition. Your task is to review the provided Python code intended to solve the competition task described below. In this first round, focus on identifying obvious bugs and potential hallucinations—such as referencing non-existent fields in the test dataset or errors in saving the submission.csv file in the './submission/' directory. Use the task details from the competition description (self.task_desc) to guide your review. In your response, first provide a brief explanation (3–5 sentences) of the identified issues and the modifications you made, and then output a single markdown code block that is an exact copy of the original code with only the minimal modifications necessary to fix these bugs. Do not change any code that is already correct.",
       "Reflection Iteration 2:Building on the code from Iteration 1, review the revised code for any residual issues. In this round, concentrate on ensuring that variable usage, data transformations, and all method imports adhere to the requirements stated in the competition description (self.task_desc). Also, check that no unintended modifications were introduced in the previous iteration. Provide a concise explanation (3–5 sentences) detailing the additional refinements you made, and then output a single markdown code block containing only the minimal modifications made relative to the Iteration 1 version.",
        ",Reflection Iteration 3:Using the code from Iteration 2 as your starting point, perform a final, comprehensive review for complete correctness and robustness. Confirm that the code fully meets the specifications from the competition description (self.task_desc), including proper data handling, consistency in method usage, and correct creation of the submission file. Provide a brief summary (3–5 sentences) of any final corrections or enhancements, and output a single markdown code block that includes only the minimal changes from the Iteration 2 code necessary to address any remaining issues."]
        current_code = code
        for i in range(reflection_steps):
            prompt = {
                "Introduction": prompts[i],
                "Original Task description": self.task_desc,
                "Previous (not revised) implementation": wrap_code(current_code),
                "Response format": (
                    "Your response should be a brief outline/sketch (3–5 sentences) explaining the modifications, "
                    "followed by a single markdown code block (wrapped in ```) that is an exact copy of the original code with only the necessary changes to fix the identified bugs."
                ),
                "Instructions": {}
            }
            reflection_plan, current_code = self.plan_and_code_query(prompt)
            # Check if the code is empty or contains only whitespace
            if not current_code.strip():
                logger.info("Reflection code is empty or contains only whitespace.")
                break
            # Check if the code is the same as the previous iteration
            if current_code == code:
                logger.info("Reflection code is the same as the previous iteration.")
                break
        return reflection_plan, current_code


    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self, exec_callback: ExecCallbackType):
        # clear the submission dir from previous steps
        shutil.rmtree(self.cfg.workspace_dir / "submission", ignore_errors=True)
        (self.cfg.workspace_dir / "submission").mkdir(exist_ok=True)

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")
        draft_flag = False
        if parent_node is None:
            draft_flag = True
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)
        # self reflection block
        # if draft_flag:
        #     reflection_plan, reflection_code = self._reflect(code=result_node.code)
        #     if reflection_code:
        #         result_node.code = reflection_code
        #         result_node.plan = reflection_plan
        #         logger.info(f"Node {result_node.id} self-reflected and updated code")
        if draft_flag:
            reflection_plan, reflection_code = self._multi_step_reflect(code=result_node.code)
            if reflection_code:
                result_node.code = reflection_code
                result_node.plan = reflection_plan
                logger.info(f"Node {result_node.id} self-reflected and updated code")
        # Proceed with execution
        logger.info(f"Agent is executing code for node {result_node.id}")
        result_node = self.parse_exec_result(node=result_node, exec_result=exec_callback(result_node.code, True))

        # handle final cases where we missed buggy nodes somehow
        if not result_node.is_buggy:
            if not (self.cfg.workspace_dir / "submission" / "submission.csv").exists():
                result_node.is_buggy = True
                result_node.metric = WorstMetricValue()
                logger.info(
                    f"Actually, node {result_node.id} did not produce a submission.csv"
                )
        self.journal.append(result_node)

        # if the result_node is the best node, cache its submission.csv and solution.py
        # to best_solution/ by copying it there
        best_node = self.journal.get_best_node()
        if best_node is not None:
            if best_node.id == result_node.id:
                logger.info(f"Node {result_node.id} is the best node so far")
                best_solution_dir = self.cfg.workspace_dir / "best_solution"
                best_solution_dir.mkdir(exist_ok=True, parents=True)
                # copy submission/submission.csv to best_submission/submission.csv
                best_submission_dir = self.cfg.workspace_dir / "best_submission"
                best_submission_dir.mkdir(exist_ok=True, parents=True)
                shutil.copy(
                    self.cfg.workspace_dir / "submission" / "submission.csv",
                    best_submission_dir,
                )
                # copy solution.py and relevant node id to best_solution/
                with open(best_solution_dir / "solution.py", "w") as f:
                    f.write(result_node.code)
                # take note of the node id of the best node
                with open(best_solution_dir / "node_id.txt", "w") as f:
                    f.write(str(result_node.id))
            else:
                logger.info(f"Node {result_node.id} is not the best node")
                logger.info(f"Node {best_node.id} is still the best node")
        self.current_step += 1

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

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
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
            ),
        )

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        # do an extra check, to catch cases where judge fails
        has_csv_submission = (
            self.cfg.workspace_dir / "submission" / "submission.csv"
        ).exists()

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
            or response["has_csv_submission"] == False
            or has_csv_submission == False
        )

        if node.is_buggy:
            logger.info(
                f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
            )
            node.metric = WorstMetricValue()
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )

        return node

import ast
import re
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Optional

import pandas as pd
from loguru import logger

from tinychat.agents.models import PandasAgentConfig
from tinychat.agents.openai_agent import OpenAIAgent
from tinychat.prompts.pandas_agent import DEFAULT_PANDAS_AGENT_PROMPT


class PandasAgent(OpenAIAgent):
    def __init__(self, config: PandasAgentConfig):
        self.df = pd.DataFrame.from_dict(config.df_dict)
        super().__init__(config=config)

    def generate_prompt(self, prompt: str) -> list[dict]:
        prompt = [
            {
                "role": "system",
                "content": DEFAULT_PANDAS_AGENT_PROMPT.format(
                    prompt=prompt,
                    df_dtypes=self.df.dtypes,
                    df_head=self.df.head(5).to_json(),
                ),
            }
        ]
        return prompt

    @staticmethod
    def calculate_accuracy(
        expected_outputs: list[pd.DataFrame], llm_outputs: list[pd.DataFrame]
    ) -> float:
        correct = 0
        total = len(expected_outputs)

        for i in range(total):
            if expected_outputs[i].equals(llm_outputs[i]):
                correct += 1

        return correct / total

    def sanitize_input(self, query: str) -> str:
        # Removes `, whitespace & python from start
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        # Removes whitespace & ` from end
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    def validate_code_safety(self, llm_code: str) -> bool:
        try:
            tree = ast.parse(llm_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    logger.debug(
                        "Generated code attempts to import, execution rejected..."
                    )
                    return False
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "open", "os", "subprocess"]:
                        logger.debug(
                            "Potentially unsafe function generated, execution rejected..."
                        )
                        return False
            return True
        except (SyntaxError, AttributeError) as e:
            logger.exception(f"Exception validating code safety: {e}")
            return False

    def execute_code(self, llm_code: str) -> Any:
        sanitized_code = self.sanitize_input(llm_code)
        if not self.validate_code_safety(sanitized_code):
            return False
        try:
            tree = ast.parse(sanitized_code)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec_globals = {"df": self.df, "pd": pd}
            exec(ast.unparse(module), exec_globals)
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)
            io_buffer = StringIO()
            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, exec_globals)
                    if ret is None:
                        return io_buffer.getvalue()
                    else:
                        return ret
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, exec_globals)
                return io_buffer.getvalue()
        except Exception as e:
            return f"{type(e).__name__}: {str(e)}"

    async def generate_code_async(self, query: str) -> str:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=self.prompt + [{"role": "user", "content": str(query)}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.exception(
                f"Error while hitting OpenAI with query: {query}",
                exc_info=True,
            )
            raise e

    async def handle_generate_response(self, query: str) -> Any:
        code = await self.generate_code_async(query)
        logger.debug(f"Code generated: {code}")
        executed = self.execute_code(code)
        if isinstance(executed, pd.DataFrame):
            return executed.head(10).to_json()
        else:
            return executed

    async def evaluate_code_generation(
        self,
        test_queries: list[str],
        expected_outputs: list[pd.DataFrame],
    ):
        llm_outputs = []
        for query in test_queries:
            generated_code = await self.generate_code_async(query)
            code_output = self.execute_code(generated_code)
            llm_outputs.append(code_output)
        accuracy = self.calculate_accuracy(expected_outputs, llm_outputs)
        return accuracy, all(output is not None for output in llm_outputs)

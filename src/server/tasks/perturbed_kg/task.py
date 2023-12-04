import sys
from typing import List, Tuple, Dict, Any

from src.server.task import Task, Session
from src.server.tasks.knowledgegraph.task import KnowledgeGraph, INSTRUCTIONS, ONE_SHOT
from src.typings import (
    TaskSampleExecutionResult,
    TaskOutput,
    SampleIndex,
    AgentOutputStatus,
    SampleStatus,
)
from .api import *
from src.server.tasks.knowledgegraph.utils.sparql_executer import SparqlExecuter


class PerturbedKnowledgeGraph(KnowledgeGraph):
    def __init__(self, data_file, sparql_url=None, round=15, **configs):
        super().__init__(
            data_file=data_file, sparql_url=sparql_url, round=round, **configs
        )
        self.item_seed = configs.pop("item_seed", 42)

    async def start_sample(
        self, index: SampleIndex, session: Session
    ) -> TaskSampleExecutionResult:
        data_item = self.inputs[index]

        answer = []
        actions = []
        variables_list = []

        question = data_item["question"]
        entities = data_item["entities"]

        session.inject({"role": "user", "content": INSTRUCTIONS})
        session.inject(
            {
                "role": "agent",
                "content": "I've understood your instruction, start please.",
            }
        )
        for idx, shot in enumerate(ONE_SHOT):
            if idx % 2 == 0:
                session.inject({"role": "user", "content": shot})
            else:
                session.inject({"role": "agent", "content": shot})
        session.inject(
            {
                "role": "user",
                "content": "A new question: "
                + question
                + "\nEntities: "
                + f"[{', '.join([entity for entity in entities])}]",
            }
        )

        finish_reason = SampleStatus.COMPLETED
        for i in range(self.round):
            message = await session.action()
            if message.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_CONTEXT_LIMIT
                )
            elif message.status != AgentOutputStatus.NORMAL:
                return TaskSampleExecutionResult(status=SampleStatus.UNKNOWN)
            message = message.content
            message = message.split("Observation:")[0]
            message = message.replace("\\_", "_")
            session.history[-1].content = message

            final_answer = re.findall(r"(?:Find|Final) Answer: #(\d+)", message)
            if final_answer:
                try:
                    answer_variable = variables_list[int(final_answer[0])]
                except IndexError:
                    session.inject(
                        {
                            "role": "user",
                            "content": "Invalid variable id! Need to recheck the action.",
                        }
                    )
                    # print({"role": "user", "content": "Invalid variable id! Need to recheck the action."})
                    continue
                answer = final_execute(answer_variable, self.sparql_executor)
                break
            else:
                lines = message.split("\n")
                find_action = False
                for line in lines:
                    execution_message = "Function is not executed!"
                    if re.match(r"Action.*?:", line):
                        find_action = True
                        function_names = re.findall(r"(\w+)\(", line)
                        function_executed = False
                        for function_name in function_names:
                            try:
                                func = getattr(sys.modules[__name__], function_name)
                                matches = re.findall(
                                    r"{}\((.+?)\)".format(function_name), line
                                )
                                arguments = re.split(r"\s*,\s*", matches[0])
                                ori_arguments = [argument for argument in arguments]
                                for i, argument in enumerate(arguments):
                                    argument = argument.replace("variable ", "")
                                    argument = argument.replace("Variable ", "")

                                    if argument.startswith("#"):
                                        # replace the variable with the actual value
                                        arguments[i] = variables_list[int(argument[1:])]
                                    elif argument in entities:
                                        arguments[i] = entities[argument]
                                arguments.append(
                                    self.sparql_executor
                                )  # add the sparql executor as the last argument
                                execution, execution_message = func(
                                    *arguments,
                                    seed=self.item_seed,
                                    do_perturbation=True,
                                )
                                actions.append(
                                    f"{function_name}({', '.join(ori_arguments)})"
                                )
                                if "##" in execution_message:
                                    # the execution message contains a variable
                                    execution_message = execution_message.replace(
                                        "##", f"#{len(variables_list)}"
                                    )
                                    variables_list.append(execution)
                                session.inject(
                                    {"role": "user", "content": execution_message}
                                )
                                function_executed = True
                                break  # at most one function is executed in one turn
                            except Exception as e:
                                try:
                                    if function_name != "intersection":
                                        execution_message = f"{function_name}({', '.join(ori_arguments)}) cannot be executed. You may make a mistake and need to fix it."
                                    else:
                                        execution_message = f"{function_name}({', '.join(ori_arguments)}) cannot be executed. The two variables are not of the same type. You may further explore them by call get_relations"
                                except UnboundLocalError:
                                    execution_message = f"I may make a syntax error when calling {function_name} (e.g., unmatched parenthesis). I need to fix it and reuse the tool"

                                continue

                        if not function_executed:
                            session.inject(
                                {"role": "user", "content": execution_message}
                            )

                        break  # should at most be one line starts with Action

                if (
                    not find_action
                ):  # only for ChatGLM-130B to make sure the conversation alternates properly
                    session.inject(
                        {
                            "role": "user",
                            "content": "No executable function found! Need to recheck the action.",
                        }
                    )
        else:
            finish_reason = SampleStatus.TASK_LIMIT_REACHED

        return TaskSampleExecutionResult(
            status=finish_reason, result={"predict": answer, "actions": actions}
        )

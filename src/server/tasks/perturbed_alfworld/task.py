from typing import Any
from src.server.task import Task, Session
from src.server.tasks.alfworld import ALFWorld
from src.server.tasks.alfworld.utils import *
from src.typings import (
    TaskOutput,
    TaskSampleExecutionResult,
    SampleStatus,
    AgentOutputStatus,
)

from .perturbations import get_items_from_obs, generate_perturbed_obs


class PerturbedALFWorld(ALFWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.data_files = self.data_files[:20]
        self.item_seed = kwargs.get("item_seed", 42)

    async def alfworld_run(self, session: Session, env: Any):
        finish_reason = SampleStatus.COMPLETED
        # env init
        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
        perturbed_obs = generate_perturbed_obs(ob, seed=self.item_seed)
        print(perturbed_obs)
        # history = self.get_prompt(name)
        # add instruction
        # history[0] = self.get_task_instruction() + "Here is one example.\n" + history[0]
        # self.inject_info(session, history)

        log_info = {"log": []}
        session.inject({"role": "user", "content": self.get_task_instruction()})
        session.inject(
            {
                "role": "agent",
                "content": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        )

        # 1-shot naive example
        history = self.get_prompt(name)
        history[0] = "Here is one example.\n" + history[0]
        self.inject_info(session, history)

        init_prompt = (
            "Here is your task. "
            + perturbed_obs
            + self.get_available_actions(info.get("admissible_commands", [[]])[0])
        )
        log_info["init_prompt"] = init_prompt
        session.inject({"role": "user", "content": init_prompt})
        # init
        # for his in session.history:
        #     print(his)
        # print("============ history end ==============")

        # interact
        for i in range(0, self.max_step):
            output = await session.action()
            if output.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                finish_reason = SampleStatus.AGENT_CONTEXT_LIMIT
                break
            output = output.content or ""

            # process action
            admissible_commands = info.get("admissible_commands", [[]])[0]
            action = process_action(output, admissible_commands)
            if not action:
                finish_reason = SampleStatus.AGENT_INVALID_ACTION
                break
            session.history[-2].content = session.history[-2].content.split(
                "AVAILABLE ACTIONS"
            )[
                0
            ]  # reduce the prompt length

            observation, reward, done, info = env.step([action])
            observation, reward, done = (
                process_ob(observation[0]),
                info["won"][0],
                done[0],
            )
            perturbed_observation = generate_perturbed_obs(observation)
            print(perturbed_observation)
            session.inject(
                {
                    "role": "user",
                    "content": perturbed_observation
                    + self.get_available_actions(
                        info.get("admissible_commands", [[]])[0]
                    ),
                }
            )

            # save
            payload = {
                "round": i + 1,
                "output": output,
                "action": action,
                "admissible_commands": admissible_commands,
                "observation": observation,
                "perturbed_observation": perturbed_observation,
                "done": done,
            }
            log_info["log"].append(payload)

            # failure test
            if len(log_info["log"]) > 3:
                pre_logs = log_info["log"][-3:]
                pre_acts = [pre_log["output"] for pre_log in pre_logs]
                if len(list(set(pre_acts))) == 1:
                    print("repeat actions for 3 times: failure")
                    return 0, log_info, SampleStatus.AGENT_INVALID_ACTION

            if done:
                return reward, log_info, finish_reason
        else:
            finish_reason = SampleStatus.TASK_LIMIT_REACHED
        return 0, log_info, finish_reason

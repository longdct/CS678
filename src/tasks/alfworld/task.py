import argparse
import sys
import json
import yaml
from tqdm.auto import tqdm

import alfworld
import alfworld.agents.environment

from src.utils.utils import set_seed
from src.utils.llm import llm

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open("base_config.yaml") as reader:
    config = yaml.safe_load(reader)

split = "eval_out_of_distribution"

env = getattr(alfworld.agents.environment, config["env"]["type"])(
    config, train_eval=split
)
env = env.init_env(batch_size=1)

folder = "./prompts/"
prompt_file = "alfworld_3prompts.json"
with open(folder + prompt_file, "r") as f:
    d = json.load(f)


def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def alfworld_run(
    prompt,
    to_print=True,
    ob="",
    max_num_duplicates=3,
    max_num_failed=3,
    temperature=0.0,
):
    init_prompt = prompt + ob + "\n>"
    prompt = ""
    if to_print:
        print(ob)
        sys.stdout.flush()
    prev_action = ""
    num_duplicates = 0
    num_failed = 0
    for i in range(1, 50):
        logger.info(f"Turn: {i}")
        action = llm(init_prompt + prompt, stop=["\n"], temperature=temperature).strip()
        # action = "go to"
        logger.info(f"Response: {action}")
        if action == prev_action:
            num_duplicates += 1
        else:
            prev_action = action
            num_duplicates = 0
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info["won"][0], done[0]
        if observation == "Nothing happens.":
            num_failed += 1
        else:
            num_failed = 0
        if action.startswith("think:"):
            observation = "OK."
        if to_print:
            print(f"Act {i}: {action}\nObs {i}: {observation}")
            sys.stdout.flush()
        logger.info(f"Act {i}: {action}\nObs {i}: {observation}")
        prompt += f" {action}\n{observation}\n>"
        if done:
            return reward
        if num_duplicates >= max_num_duplicates or num_failed >= max_num_failed:
            break
    return 0


def main(args):
    set_seed(args.seed)
    handler = logging.FileHandler(args.log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    prefixes = {
        "pick_and_place": "put",
        "pick_clean_then_place": "clean",
        "pick_heat_then_place": "heat",
        "pick_cool_then_place": "cool",
        "look_at_obj": "examine",
        "pick_two_obj": "puttwo",
    }
    cnts = [0] * 6
    rs = [0] * 6

    for _ in range(100):
        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
        logger.info(f"Env name: {name}")
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):  # Matching env with specific prompt
                prompt = (
                    "Interact with a household to solve a task. Here are two examples.\n"
                    + d[f"react_{v}_1"]
                    + d[f"react_{v}_0"]
                    + "\nHere is the task.\n"
                )
                r = alfworld_run(
                    prompt, ob=ob, to_print=False, temperature=args.temperature
                )
                rs[i] += r
                cnts[i] += 1
                break
        logger.info(
            "Results: Episode success {} All success {} Total {} Percentage {}".format(
                r,
                rs,
                cnts,
                sum(rs) / sum(cnts),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=233, help="Random seed")
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="temperature",
    )
    parser.add_argument("--log_path", type=str, default="alfworld_gpt-3.5.log")
    args = parser.parse_args()
    main(args)

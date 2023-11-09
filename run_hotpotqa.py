import argparse
import json
import random
import requests
import time
from tqdm.auto import tqdm

import llm
import wikienv, wrappers
from utils import set_seed
from llm import llm

import logging

logger = logging.getLogger("hotpotqa")
logger.setLevel(logging.INFO)

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

folder = "./prompts/"
prompt_file = "prompts_naive.json"
with open(folder + prompt_file, "r") as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict["webthink_simple6"]
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)

        except requests.exceptions.Timeout:
            attempts += 1


def webthink(idx=None, prompt=webthink_prompt, to_print=True, temperature=0.0):
    question = env.reset(idx=idx)
    logger.info(f"Index {idx} Question: {question}")
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        logger.info(f"Turn: {i}")
        n_calls += 1
        thought_action = llm(
            prompt + f"Thought {i}:",
            stop=[f"\nObservation {i}:"],
            temperature=temperature,
        )
        logger.info(f"Response: {thought_action}")
        if thought_action is None:
            thought_action = ""
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print("ohh...", thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            action = llm(
                prompt + f"Thought {i}: {thought}\nAction {i}:",
                stop=[f"\n"],
                temperature=temperature,
            ).strip()
        logger.info(f"Action: {action}")
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace("\\n", "")
        step_str = (
            f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        )
        prompt += step_str
        if to_print:
            print(step_str)
            print(prompt)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, "\n")
    info.update({"n_calls": n_calls, "n_badcalls": n_badcalls})
    logger.info(info)
    info.update({"traj": prompt})
    return r, info


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
    parser.add_argument("--log_path", type=str, default="hotpotqa_gpt-3.5.log")
    args = parser.parse_args()

    set_seed(args.seed)
    handler = logging.FileHandler(args.log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    idxs = list(range(7405))
    random.Random(args.seed).shuffle(idxs)

    logger.info("RUN AGENT!")
    rs = []
    infos = []
    old_time = time.time()
    for i in tqdm(idxs[:100]):
        r, info = webthink(i, to_print=False, temperature=args.temperature)
        rs.append(info["em"])
        infos.append(info)
        logger.info(
            "Results: Success {} Total {} Percentage {} Time {}".format(
                sum(rs),
                len(rs),
                sum(rs) / len(rs),
                (time.time() - old_time) / len(rs),
            )
        )

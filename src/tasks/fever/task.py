import argparse
import json
import random
import time
from tqdm.auto import tqdm

import requests

from src.environments import wikienv, wrappers
from src.utils.utils import set_seed
from src.utils.llm import llm
from src.adversarial_attack import ATTACK_TYPES, create_attack
from src.multilingual import Translator

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

env = wikienv.WikiEnv()
env = wrappers.FeverWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)


folder = "./prompts/"
prompt_file = "fever.json"
with open(folder + prompt_file, "r") as f:
    prompt_dict = json.load(f)

webthink_prompt = prompt_dict["webthink_simple3"]


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1


def webthink(
    idx=None,
    prompt=webthink_prompt,
    to_print=True,
    augmenter=None,
    augment_setting=None,
    temperature=0.0,
):
    question = env.reset(idx=idx)
    if augment_setting in ["cross", "multi"]:
        question = augmenter.augment(question)[0]
    logger.info(f"Index {idx} Question: {question}")
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        logger.info(f"Turn: {i}")
        n_calls += 1
        # thought_action = f"Think\nAction {i}: Act"
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
        logger.info(f"Original observation: {obs}")
        if augmenter is not None:
            obs = augmenter.augment(obs)[0]
            logger.info(f"Adversarial observation: {obs}")
        step_str = (
            f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        )
        prompt += step_str
        if to_print:
            print(step_str)
            print(prompt)
        # logger.info(step_str)
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


def main(args):
    set_seed(args.seed)
    handler = logging.FileHandler(args.log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("SETUP AUGMENTATION!")
    if args.perturb_type.startswith("translate"):
        _, target_lang, augment_setting = args.perturb_type.split("_")
        augmenter = Translator(target_lang)

        global webthink_prompt

        if augment_setting == "multi":
            webthink_prompt = prompt_dict["webthink_simple3_multi_de"]
        elif augment_setting == "cross":
            webthink_prompt += "\n\nThe examples given to you above is in English. Now, the question and observation will be in "
            if target_lang == "de":
                webthink_prompt += "Germans."
            elif target_lang == "hi":
                webthink_prompt += "Hindi."
            elif target_lang == "et":
                webthink_prompt += "Estonian."
            webthink_prompt += " You must use the same action types (i.e., SEARCH, LOOKUP, FINISH) in English and the same Thought-Action format to finish this task. You MUST NOT generate actions in other languages.\n\n"
        print(augment_setting, webthink_prompt)
    elif args.perturb_type != "none":
        augmenter = create_attack(
            args.perturb_type,
            pct_words_to_swap=args.pct_words_to_swap,
            transformations_per_example=1,
        )
        augment_setting = "adversarial"
    else:
        augmenter = None
        augment_setting = None

    logger.info("RUN AGENT!")
    idxs = list(range(7405))
    random.Random(args.seed).shuffle(idxs)
    blacklisted = [2823, 3188]
    for idx in blacklisted:
        idxs.remove(idx)
    idxs = idxs[:100]
    rs = []
    infos = []
    old_time = time.time()
    for i in tqdm(idxs):
        if i in blacklisted:
            continue
        try:
            r, info = webthink(
                i,
                to_print=False,
                prompt=webthink_prompt,
                augmenter=augmenter,
                augment_setting=augment_setting,
                temperature=args.temperature,
            )
        except Exception as e:
            logger.warning(e)
            info = {
                "steps": -1,
                "answer": None,
                "gt_answer": None,
                "question_idx": i,
                "em": 0,
                "reward": 0,
                "f1": 0,
                "n_calls": -1,
                "n_badcalls": -1,
            }
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=233, help="Random seed")
    parser.add_argument(
        "--perturb_type",
        "-a",
        type=str,
        default="none",
        choices=ATTACK_TYPES
        + ["none"]
        + [
            f"translate_{lang}_{setting}"
            for lang in ["de", "hi", "et"]
            for setting in ["cross", "multi"]
        ],
        help="Perturbation type",
    )
    parser.add_argument(
        "--pct_words_to_swap",
        "-p",
        type=float,
        default=0.1,
        help="Percentage words to swap",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="temperature",
    )
    parser.add_argument("--log_path", type=str, default="perturbed_gpt-3.5.log")
    args = parser.parse_args()
    main(args)

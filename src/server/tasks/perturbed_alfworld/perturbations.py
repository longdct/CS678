from typing import List, Dict

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    # Set seed for random module
    random.seed(seed)

    # Set seed for numpy module
    np.random.seed(seed)

    # Set seed for torch module
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # print(f"Seeds set to {seed} for random, numpy, and torch.")


def get_items_from_info(info: Dict) -> List[str]:
    admissible_commands = info.get("admissible_commands")[0]
    items_commands = filter(lambda s: "go to" in s, admissible_commands)
    items = map(lambda s: s.replace("go to", "").strip(), items_commands)
    return sorted(list(items))


def extract_item_str_from_obs(obs: str) -> str:
    obs = obs.strip().split("\n")[0]
    signal_str = "you see "
    if signal_str not in obs:
        return ""
    idx = obs.index(signal_str)
    item_str = obs[idx + len(signal_str) :]
    return item_str


def get_items_from_obs(obs: str) -> List[str]:
    def extract_item(s):
        s = s.strip().replace(".", "")
        if s.startswith("and"):
            s = s.replace("and", "").strip()
        if s.startswith("a "):
            s = s.replace("a ", "").strip()
        return s

    item_str = extract_item_str_from_obs(obs)
    items = map(extract_item, item_str.split(", "))
    return sorted(list(items))


def generate_perturbed_obs(obs: str, seed: int = 42) -> str:
    item_str = extract_item_str_from_obs(obs)
    if not item_str:
        return obs  # Observation doesn't contain any items
    idx_item_str = obs.index(item_str)
    sub_str1 = obs[:idx_item_str]
    sub_str2 = obs[idx_item_str + len(item_str) :]

    set_seed(seed)
    items = get_items_from_obs(obs)
    random.shuffle(items)
    perturbed_obs = sub_str1
    for i, item in enumerate(items):
        perturbed_obs += " a " + item
        if i < len(items) - 1:
            perturbed_obs += ","
    perturbed_obs += "."
    perturbed_obs += "\n" + sub_str2
    return perturbed_obs

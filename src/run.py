import argparse
from src.adversarial_attack import ATTACK_TYPES
from src.multilingual import SUPPORTED_LANGUAGES


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["alfworld", "hotpotqa", "fever"],
        help="Task to run",
    )
    parser.add_argument("--seed", type=int, default=233, help="Random seed")
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="temperature",
    )
    parser.add_argument("--log_path", type=str, default="gpt-3.5.log")
    parser.add_argument(
        "--perturb_type",
        type=str,
        default="none",
        choices=ATTACK_TYPES
        + ["none"]
        + [
            f"translate_{lang}_{setting}"
            for lang in SUPPORTED_LANGUAGES
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
    args = parser.parse_args()

    if args.task == "fever":
        from src.tasks.fever.task import main

        main(args)
    elif args.task == "alfworld" and args.perturb_type == "none":
        from src.tasks.alfworld.task import main

        main(args)
    elif args.task == "hotpotqa" and args.perturb_type == "none":
        from src.tasks.hotpotqa.task import main

        main(args)
    else:
        raise NotImplementedError("Task is not implemented")

#! /bin/bash

for task in fever alfworld hotpotqa; do
    python -m src.run \
        --task ${task} \
        --temperature 0.0 \
        --log_path logs/${task}.gpt-35.log \
        --perturb_type none
done

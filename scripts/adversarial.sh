#! /bin/bash

for attack in textbugger textfooler checklist; do
    python -m src.run \
        --task fever \
        --temperature 0.0 \
        --log_path logs/fever.${attack}.gpt-35.log \
        --perturb_type ${attack} \
        --pct_words_to_swap 0.5
done

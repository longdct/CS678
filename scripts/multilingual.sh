#! /bin/bash

SETTING=multi
for lang in de; do
    python -m src.run \
        --task fever \
        --temperature 0.0 \
        --log_path logs/fever.${SETTING}_${lang}.gpt-35.log \
        --perturb_type translate_${lang}_${SETTING}
done

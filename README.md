# CS678

Reproduce ICLR 2023 paper [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).

## Setup
You need to first have an OpenAI API key and store it in the environment variable ``OPENAI_API_KEY``.

Package requirement: 
  - ``openai==1.0.1``
  - ``tqdm``
  - ``alfworld``: following instructions [here](https://github.com/alfworld/alfworld).
  - ``transformers==4.33.1``
  - ``textattack``: following instructions [here](https://github.com/QData/TextAttack).

## Experiments
Scripts to run experiments are included in `scripts` folder. In particular:
  - To reproduce results from ReAct paper, run script `reproduce.sh`
  - To run adversarial robustness experiment, run script `adversarial.sh`
  - To run cross-lingual experiment, run script `crosslingual.sh`
  - To run multi-lingual experiment, run script `multilingual.sh`

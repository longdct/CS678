# CS678

Reproduce of ICLR 2023 paper [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).

## Setup
You need to first have an OpenAI API key and store it in the environment variable ``OPENAI_API_KEY``.

Package requirement: 
  - ``openai==1.0.1``
  - ``tqdm``
  - ``alfworld``: following instructions [here](https://github.com/alfworld/alfworld).

## Experiments
Run ``{run_hotpotqa,run_fever,run_alfworld}.py`` to run each dataset. Reproduced results are found in `logs` folder.  


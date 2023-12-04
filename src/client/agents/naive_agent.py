from typing import List, Dict

from ..agent import AgentClient


class NaiveAgent(AgentClient):
    def __init__(self, text: str, *args, **config):
        super().__init__(*args, **config)
        self.text = text

    def inference(self, history: List[dict]) -> str:
        # print(history)
        return self.text

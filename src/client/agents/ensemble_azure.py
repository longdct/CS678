from collections import Counter
from openai import AzureOpenAI
from typing import List, Dict

from .azure import Azure


class EnsembleAzure(Azure):
    def inference(self, history: List[dict]) -> str:
        prompt = Azure._process_history()(history)
        c = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            **self.client_args,
        )
        resp = c.chat.completions.create(messages=prompt, **self.api_args)
        sampled_outputs = [choice.message.content for choice in resp.choices]
        c = Counter(sampled_outputs)
        return c.most_common()[0][0]

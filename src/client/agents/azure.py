from openai import AzureOpenAI
import os
from copy import deepcopy
from typing import List, Dict

from ..agent import AgentClient


class Azure(AgentClient):
    def __init__(self, client_args=None, api_args=None, *args, **config):
        super().__init__(*args, **config)
        if not client_args:
            client_args = {}
        client_args = deepcopy(client_args)
        if not api_args:
            api_args = {}
        api_args = deepcopy(api_args)

        self.client_args = client_args
        self.api_key = client_args.pop("api_key", None) or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API KEY is required, please assign client_args.key or set OPENAI_API_KEY "
                             "environment variable.")
        self.azure_endpoint = client_args.pop("endpoint", None)
        if not self.azure_endpoint:
            raise ValueError("Azure endpoint is required, please assign client_args.azure_endpoint.")
              
        self.api_args = api_args
        api_args["model"] = api_args.pop("model", None)
        if not api_args["model"]:
            raise ValueError("OpenAI model is required, please assign api_args.model.")
        if not self.api_args.get("stop"):
            self.api_args["stop"] = ["\n"]

    @staticmethod
    def _process_history(
        message_key: str = "messages",
        role_key: str = "role",
        content_key: str = "content",
        user_role: str = "user",
        agent_role: str = "assistant",
    ):
        def prompter(messages: List[Dict[str, str]]):
            nonlocal message_key, role_key, content_key, user_role, agent_role
            role_dict = {
                "user": user_role,
                "agent": agent_role,
            }
            prompt = []
            for item in messages:
                prompt.append(
                    {role_key: role_dict[item["role"]], content_key: item["content"]}
                )
            return prompt

        return prompter

    def inference(self, history: List[dict]) -> str:
        prompt = Azure._process_history()(history)
        c = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            **self.client_args,
        )
        resp = c.chat.completions.create(messages=prompt, **self.api_args)
        return resp.choices[0].message.content

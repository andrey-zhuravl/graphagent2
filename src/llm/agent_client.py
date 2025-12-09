from typing import Iterable

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.utils.config import get_config_dict

class AgentClient:
    def __init__(self):
        self.config = get_config_dict()
        self.client = OpenAI(base_url=self.config["llm"]["base_url"], api_key="none")

    def request(self, msgs: Iterable[ChatCompletionMessageParam] = None, prompt: str = None):
        if not msgs:
            msgs = [{"role": "user", "content": prompt}]
        result =  self.client.chat.completions.create(
            model=self.config["llm"]["model"],
            messages=msgs,
            temperature=0.0,
            max_tokens=8192,
            stop=["\n```"]  # обрезаем после первого ```
        )
        return result
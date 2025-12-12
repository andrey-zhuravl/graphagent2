import json
from typing import List

from src.action import Action
from src.llm.agent_client import AgentClient
from src.mcp_server.mcp_client import McpClient
from src.memory import Context, Thought

class Policy:
    def __init__(self, client: AgentClient = None, mcp_client: McpClient = None):
        self.client = AgentClient()  # URL для запроса к LLM
        self.mcp_client = McpClient()  # URL для получения списка инструментов

    def get_tools_list(self) -> dict[str, list[str]]:
        """
        Получаем список доступных инструментов через API.
        """
        try:
            categories = self.mcp_client.get_tool_categories()
            tools_list = self.mcp_client.dict_tools(categories)
            return tools_list
        except Exception as e:
            print(f"Error fetching tools: {e}")
            return {}

    def generate_prompt(self, tools: List[str], ctx: Context) -> str:
        """
        Формируем промт для LLM на основе доступных инструментов и контекста.
        """
        tool_names = ", ".join(tools)
        history_text = "\n".join(
            [f"Action: {obs.action.tool_name}({obs.action.params})\nOutput: {obs.output}\nSuccess: {obs.success}" for
             obs in ctx.memory.history])
        prompt = f"""
        You are an RaAct agent controlling tools:
        {tool_names}
        
        Decide the next action.
        
        User goal: {ctx.user_goal}

        History:
        {history_text}
        
        Please select the best tool and provide the necessary parameters for the action.
        Return ONLY a JSON object:
        {{
          "tool_name": "...",
          "params": {{ ... }}
        }}
        """
        return prompt

    def select_action(self, ctx: Context) -> Action:
        """
        Вызывает LLM, чтобы выбрать действие и инструмент.
        """
        # Получаем список доступных инструментов
        tools = self.get_tools_list()
        tools_list = tools["filesystem"] + tools["git"]

        # Формируем промт для LLM
        prompt = self.generate_prompt(tools_list, ctx)

        action = self.get_next_action_from_llm(ctx, prompt)
        return action

    def get_next_action_from_llm(self, ctx: Context, prompt):
        # Отправляем запрос к LLM с промтом
        try:
            response = self.client.request(
                msgs=[{"role": "user", "content": prompt}]
            )
            llm_output = response.choices[0].message.content.strip()

            # Извлекаем выбранный инструмент и параметры
            tool_name = json.loads(llm_output).get("tool_name")
            params = json.loads(llm_output).get("params", {})

            # if tool_name not in tools:
            #     raise ValueError(f"Tool {tool_name} is not in the available tools list.")

            return Action(tool_name=tool_name, params=params)

        except Exception as e:
            print(f"Error calling LLM: {e}")
            return Action(tool_name="done", params={})  # Если ошибка, завершаем действие

# ---- Пример использования ----
#
# # Настроим агента с использованием URL для вашего LLM и списка инструментов
# policy = LLMPolicy(
#     llm_url="http://127.0.0.1:8000/llm",  # Это пример, подставьте ваш URL
#     tools_api_url="http://127.0.0.1:8000/mcp"  # API для получения списка инструментов
# )
#
# # Запускаем агент с заданным контекстом и политикой
# context = Context(memory=Memory(), target_class=SomeClass, user_goal="Generate a test case")
# result = run_react(ctx=context, policy=policy, tools=tools_dict)

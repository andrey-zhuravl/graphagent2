import asyncio

from src.agent import Agent


async def main():
    agent = Agent()
    task1 = "Напиши создай файл с рассказом про кота и собаку story.py"
    await agent.async_run(task=task1)


if __name__ == "__main__":
    asyncio.run(main())
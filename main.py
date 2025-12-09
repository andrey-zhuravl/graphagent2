from src.agent import Agent


def main():
    agent = Agent()
    task1 = "Напиши создай файл с рассказом про кота и собаку story.py"
    agent.run(task=task1)


if __name__ == "__main__":
    main()
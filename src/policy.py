from src.action import Action
from src.memory import Context


class Policy:
    def select_action(self, ctx: Context) -> Action:
        # 1) Если ещё не сгенерировали тесты — вызываем генератор
        if not ctx.memory.scratchpad.get("tests_generated"):
            return Action(
                tool_name="code_generator",
                params={"target": ctx.target_class}
            )

        # 2) Если тесты сгенерированы, но не записаны — пишем файл
        if not ctx.memory.scratchpad.get("tests_written"):
            gen_test = ctx.memory.scratchpad["tests_generated"]
            return Action(
                tool_name="filesystem_write",
                params={
                    "path": gen_test["path"],
                    "content": gen_test["content"]
                }
            )

        # 3) Если тесты записаны — запускаем runner
        if not ctx.memory.scratchpad.get("tests_ran"):
            return Action(
                tool_name="test_runner",
                params={"path": ctx.memory.scratchpad["tests_generated"]["path"]}
            )

        # 4) Всё сделано — агент заканчивает
        return Action(tool_name="done", params={})

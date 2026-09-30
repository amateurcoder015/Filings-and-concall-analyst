from types import SimpleNamespace as NS


def tool_use(name, input, id="t1"):
    return NS(type="tool_use", id=id, name=name, input=input)


def text(value):
    return NS(type="text", text=value)


def response(*blocks):
    return NS(content=list(blocks), stop_reason="tool_use")


class ScriptedClient:
    """Stands in for anthropic.Anthropic: replays a script of responses or exceptions."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        kwargs["messages"] = list(kwargs["messages"])
        self.calls.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

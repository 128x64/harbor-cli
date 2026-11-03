"""Fake llama-cpp-python used by the tests: speaks the Hermes <tool_call> protocol."""
class Llama:
    def __init__(self, **kw):
        self.kw = kw

    def create_chat_completion(self, messages, max_tokens=None, temperature=None, stream=False):
        last = messages[-1]["content"]
        if "<tool_response>" in last:
            parts = ["All done ", "via local."]
        elif "<tools>" in messages[0]["content"]:
            parts = ["Creating.\n<tool_", 'call>\n{"name": "write_file", "arguments": {"path": "local.txt", "content": "hi"}}\n</tool_call>']
        else:
            parts = ["summary"]
        for p in parts:
            yield {"choices": [{"delta": {"content": p}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

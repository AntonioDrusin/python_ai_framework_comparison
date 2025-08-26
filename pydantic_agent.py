import re
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Model
provider = OpenAIProvider(base_url="http://localhost:1234/v1", api_key="not_required")
model = OpenAIChatModel("qwen/qwen3-4b", provider=provider)

agent = Agent(
    model,
    system_prompt=(
        "TOOLS ARE AUTHORITATIVE. If you call a tool and it returns a result, "
        "do NOT verify, recompute, or override it. Return the tool's result directly "
        "(with minimal formatting). Only add reasoning if the user explicitly asks. "
        "If no tool is used, answer normally."
    ),
)

# Tools
@agent.tool_plain
def calculator(expr: str) -> str:
    """Evaluates the expression in python and returns the result."""
    return str(eval(expr))
 
# Run
result = agent.run_sync("What is the result of 123*456 but talk like a pirate")

# Strip any <think>...</think> sections if the model emits them
clean = re.sub(r"<think>.*?</think>", "", result.output, flags=re.DOTALL | re.IGNORECASE).strip()

print("RESPONSE")
print(clean)
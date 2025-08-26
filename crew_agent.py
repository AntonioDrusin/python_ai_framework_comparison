import re
from typing import Type
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# --- LLM (OpenAI-compatible local endpoint) ---
llm = LLM(
    model="gpt-3.5-turbo",                 # your served model id
    base_url="http://localhost:1234/v1",   # OpenAI-compatible API
    api_key="not_required",
    temperature=0
)

# --- Calculator Tool (authoritative; returns final answer) ---

class CalcInput(BaseModel):
    expr: str = Field(..., description="A valid Python expression to evaluate.")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluates the expression in python and returns the result."
    args_schema: Type[BaseModel] = CalcInput

    def _run(self, expr: str) -> str:
        return str(eval(expr))

calculator = CalculatorTool(result_as_answer=True)  # force tool output as final

# --- Agent ---
agent = Agent(
    role="Math Answerer",
    goal="Return exact tool outputs when a tool is used; only add reasoning if explicitly requested.",
    backstory=("TOOLS ARE AUTHORITATIVE. If you call a tool and it returns a result, "
               "do NOT verify, recompute, or override it. "
               "(with minimal formatting). If no tool is used, answer normally."),
    llm=llm,
    tools=[calculator],
    verbose=True
)

# --- Task & Crew ---
task = Task(
    description="What is the result of 123*456 but talk like a pirate",
    agent=agent,
    expected_output=""
)

crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
result = crew.kickoff()  # CrewOutput

# --- Optional: strip any <think>...</think> the model might emit ---
clean = re.sub(r"<think>.*?</think>", "", (result.raw or str(result)), flags=re.DOTALL | re.IGNORECASE).strip()

print("RESPONSE")
print(clean)

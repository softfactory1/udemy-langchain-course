import re
import inspect
from typing import Callable

import ollama
from loguru import logger

from phoenix.otel import register
from settings import SETTINGS

PRICES = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
DISCOUNT_PERCENTAGES = {"bronze": 5, "silver": 12, "gold": 23}
MAX_ITERATIONS = 10

tracer_provider = register(
    project_name=SETTINGS.PHOENIX_PROJECT_NAME,     # project name shown in Phoenix
    endpoint=SETTINGS.PHOENIX_TRACES_ENDPOINT,    # OTLP over HTTP
    auto_instrument=SETTINGS.PHOENIX_AUTO_INSTRUMENT,  # picks up openinference-instrumentation-langchain
    batch=True,
)
tracer = tracer_provider.get_tracer(__name__)


@tracer.tool(name="get_product_price")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    logger.info(f"    >> Executing get_product_price(product='{product}')")
    return PRICES.get(product, 0)


@tracer.tool(name="apply_discount")
def apply_discount(price: float | str, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    logger.info(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount = DISCOUNT_PERCENTAGES.get(discount_tier, 0)
    return round(float(price) * (1 - discount / 100), 2)


def get_tool_descriptions(tools_dict: dict[str, Callable]) -> str:
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


@tracer.llm(name="Ollama chat")
def ollama_chat(messages, options):
    return ollama.chat(model=SETTINGS.OLLAMA_MODEL, messages=messages, options=options)


def get_tool_observation(tools: dict[str, Callable], output: str) -> str | None:
    logger.info(f"  [Parsing] Looking for Action and Action Input in LLM output...")
    action_match = re.search(r"Action:\s*(.+)", output)
    action_input_match = re.search(r"Action Input:\s*(.+)", output)

    if not action_match or not action_input_match:
        logger.error("  [Parsing] ERROR: Could not parse Action/Action Input from LLM output")
        return None
    tool_name = action_match.group(1).strip()
    tool_input_raw = action_input_match.group(1).strip()

    logger.info(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

    # Split comma-separated args; strip key= prefix if LLM outputs key=value format
    raw_args = [x.strip() for x in tool_input_raw.split(",")]
    args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

    logger.info(f"  [Tool Executing] {tool_name}({args})...")
    if tool_name in tools:
        return str(tools[tool_name](*args))
    logger.error(f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}")
    return None


def get_final_answer(output: str) -> str | None:
    logger.info(f"  [Parsing] Looking for Final Answer in LLM output...")
    final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
        logger.info("=" * 60)
        logger.info(f"Final Answer: {final_answer}")
        return final_answer
    return None


@tracer.agent(name="LangChain Agent Loop")
def run_agent(question: str):
    """Run the agent."""
    logger.info(f"    >> Executing run_agent(question='{question}')")
    tools: list[Callable] = [get_product_price, apply_discount]
    tools_dict: dict[str, Callable] = {t.__name__: t for t in tools}

    tool_descriptions = get_tool_descriptions(tools_dict)
    tool_names = ", ".join(tools_dict.keys())

    react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
    """

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS+1):
        logger.info(f"    >> Executing run_agent(iteration={iteration})")
        full_prompt = prompt + scratchpad
        response = ollama_chat(
            [{"role": "user", "content": full_prompt}],
            {"stop": ["\nObservation"], "temperature": 0}
        )
        output = response.message.content

        if final_answer := get_final_answer(output):
            return final_answer

        if observation := get_tool_observation(tools_dict, output):
            scratchpad += f"{output}\nObservation: {observation}\nThought:"

    logger.error("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    logger.info("Hello LangChain Agent (.bind_tools) [STARTED]")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    logger.info(f"Final result: {result}")
    logger.info("Hello LangChain Agent (.bind_tools) [FINISHED]")

from typing import Callable

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import BaseTool
from loguru import logger

from phoenix.otel import register
from settings import SETTINGS

PRICES = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
DISCOUNT_PERCENTAGES = {"bronze": 5, "silver": 12, "gold": 23}
MAX_ITERATIONS = 10
SYSTEM_MESSAGE = (
    "You are a helpful shopping assistant. "
    "You have access to a product catalog tool "
    "and a discount tool.\n\n"
    "STRICT RULES — you must follow these exactly:\n"
    "1. NEVER guess or assume any product price. "
    "You MUST call get_product_price first to get the real price.\n"
    "2. Only call apply_discount AFTER you have received "
    "a price from get_product_price. Pass the exact price "
    "returned by get_product_price — do NOT pass a made-up number.\n"
    "3. NEVER calculate discounts yourself using math. "
    "Always use the apply_discount tool.\n"
    "4. If the user does not specify a discount tier, "
    "ask them which tier to use — do NOT assume one.\n"
    f"5. The following products are available:\n {", ".join(PRICES.keys())}\n"
)

tracer_provider = register(
    project_name=SETTINGS.PHOENIX_PROJECT_NAME,     # project name shown in Phoenix
    endpoint=SETTINGS.PHOENIX_TRACES_ENDPOINT,    # OTLP over HTTP
    auto_instrument=SETTINGS.PHOENIX_AUTO_INSTRUMENT,  # picks up openinference-instrumentation-langchain
    batch=True,
)
tracer = tracer_provider.get_tracer(__name__)


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    logger.info(f"    >> Executing get_product_price(product='{product}')")
    return PRICES.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    logger.info(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount = DISCOUNT_PERCENTAGES.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


@tracer.agent(name="LangChain Agent Loop")
def run_agent(question: str):
    """Run the agent."""
    logger.info(f"    >> Executing run_agent(question='{question}')")
    tools: list[BaseTool] = [get_product_price, apply_discount]
    tools_dict: dict[str, BaseTool] = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{SETTINGS.OLLAMA_MODEL}", temperature=0.0, reasoning=True)
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_MESSAGE),
        HumanMessage(content=question),
    ]
    for iteration in range(1, MAX_ITERATIONS+1):
        logger.info(f"    >> Executing run_agent(iteration={iteration})")

        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)
        if reasoning := ai_message.additional_kwargs.get("reasoning_content"):
            with tracer.start_as_current_span(
                f"Reasoning {iteration}",
                openinference_span_kind="chain",
            ) as span:
                logger.info(f"    >> Reasoning: {reasoning}")
                span.set_attribute("reasoning", reasoning)
        if tool_calls := ai_message.tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                if tool_handler := tools_dict.get(tool_name):
                    observation = tool_handler.invoke(tool_call.get("args", {}))
                    logger.info(f"    >> Tool {tool_name} is executed, result: {observation}")
                    tool_call_id = tool_call.get("id")
                    messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
                    continue
                logger.error(f"    >> Tool {tool_name} is NOT executed")
            continue
        return ai_message.content
    logger.error("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    logger.info("Hello LangChain Agent (.bind_tools) [STARTED]")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    logger.info(f"Final result: {result}")
    logger.info("Hello LangChain Agent (.bind_tools) [FINISHED]")

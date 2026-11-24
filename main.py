from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from loguru import logger
import httpx
from langchain_tavily import TavilySearch

from phoenix.otel import register
from pyserxng import LocalSearXNGClient
from llm.ollama import OllamaManager

from langchain.agents.structured_output import ProviderStrategy
from llm.utils import supports_structured_output
from schemas import AgentResponse
from settings import SETTINGS
from utils import SearXNGClient



tracer_provider = register(
    project_name=SETTINGS.PHOENIX_PROJECT_NAME,     # project name shown in Phoenix
    endpoint=SETTINGS.PHOENIX_TRACES_ENDPOINT,    # OTLP over HTTP
    auto_instrument=SETTINGS.PHOENIX_AUTO_INSTRUMENT,  # picks up openinference-instrumentation-langchain
    batch=True,
)



#searxng_client = SearXNGClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     logger.info(f"Searching for query {query}")
#     return searxng_client.search(query)
#     # with LocalSearXNGClient(SETTINGS.SEARX_HOST) as client:
#     #     if client.test_connection():
#     #         results = client.search(query)
#     #         contents = [r.content for r in results.results[:SETTINGS.SEARXNG_MAX_RESULTS]]
#     #         return "\n".join(contents)
#     #     raise Exception("Could not connect to SearXNG")
#
#     # resp = httpx.get(
#     #     f"{SETTINGS.SEARX_HOST}/search",
#     #     params={"q": query, "format": "json"},
#     #     timeout=15.0,
#     # )
#     # resp.raise_for_status()
#     # results = resp.json().get("results", [])[: SETTINGS.SEARXNG_MAX_RESULTS]
#     # if not results:
#     #     return "No results found."
#     # return "\n\n".join(
#     #     f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')}"
#     #     for r in results
#     # )


def main():
    logger.info("Hello from udemy-langchain-course [STARTED]!")

    manager = OllamaManager(model_name=SETTINGS.OLLAMA_MODEL, temperature=0.0)
    llm = manager.get_chat()
    # llm = ChatOllama(model=SETTINGS.OLLAMA_MODEL, temperature=0.0)
    # #llm._client = Val
    # try:
    #     validate_model(client=llm._client, model_name=SETTINGS.OLLAMA_MODEL_NAME)
    # except Exception as exc:
    #     logger.error(f"Access Ollama failure {exc}")
    #     return
    #tools = [search]

    #ok = supports_structured_output(manager.openai_client, SETTINGS.OLLAMA_MODEL)

    tools = [TavilySearch(tavily_api_key=SETTINGS.TAVILY_API_KEY, max_results=3)]
    agent = create_agent(
        model=llm,
        tools=tools,
        #response_format=ProviderStrategy(AgentResponse),
        response_format=AgentResponse,
    )
    #question = "What is the capital of Japan?"
    #question = "What's the weather in Tokyo?"
    #question = "What’s the population of Tokyo? Year 2026"
    #question = "What is 2 + 2. Do not use search tool"
    #question = "Answer from your own knowledge only, without calling any tools. What is 2 + 2?"
    #question = "Reply with just the number, no tool calls: what is 2 + 2?"
    question = (
        "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin\n"
        " and list their details? when using tavily_search tool, query parameter for this toll should be always present"
    )
    result = agent.invoke(input={"messages": HumanMessage(content=question)})
    ok = 1
    #logger.info(result)
    for message in result["messages"]:
        # if not message.content:
        #     continue
        # if hasattr(message, "tool_calls"):
        #     continue
        logger.info(f"{message.type.upper()}: {message.content} [tool_call: {hasattr(message, "tool_calls")}]")
    #searxng_client.close()
    logger.info("Hello from udemy-langchain-course [FINISHED]!")


if __name__ == "__main__":
    main()

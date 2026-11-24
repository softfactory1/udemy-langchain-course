import json
from loguru import logger
from openai import OpenAI


def supports_structured_output(client: OpenAI, model: str) -> bool:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 1+1?"}],
            response_format={"type": "json_schema", "json_schema": {"name": "probe", "schema": schema, "strict": True}},
            max_tokens=50,
        )
        data = json.loads(r.choices[0].message.content)
        return isinstance(data.get("answer"), int) and set(data) == {"answer"}
    except Exception as e:
        logger.error(f"Structured output not supported for model {model}, exception: {e}")
        return False

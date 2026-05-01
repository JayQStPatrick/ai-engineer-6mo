import litellm
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
async def call_llm(messages, model="gpt-4o-mini", stream=False, response_format=None):
    return await litellm.acompletion(
        model=model,
        messages=messages,
        stream=stream,
        response_format=response_format,
        timeout=30,
    )

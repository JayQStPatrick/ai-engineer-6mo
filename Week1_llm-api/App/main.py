import json

import litellm
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.llm import call_llm

app = FastAPI()


class ChatRequest(BaseModel):
    messages: list[dict]
    stream: bool = False


class ExtractRequest(BaseModel):
    text: str
    fields: list[str] = ["name", "date", "amount"]

class SummariseRequest (BaseModel):
    text : str



@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.stream:

        async def stream_tokens():
            resp = await call_llm(req.messages, stream=True)
            async for chunk in resp:
                token = chunk.choices[0].delta.content or ""
                yield f"data: {json.dumps({'token': token})}\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")
    resp = await call_llm(req.messages)
    return {"content": resp.choices[0].message.content}


@app.post("/extract")
async def extract(req: ExtractRequest):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "extraction",
            "schema": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in req.fields},
                "required": req.fields,
            },
            "strict": True,
        },
    }
    resp = await call_llm(
        messages=[
            {"role": "user", "content": f"Extract {req.fields} from: {req.text}"}
        ],
        response_format=schema,
    )
    return json.loads(resp.choices[0].message.content)

@app.post("/summarise")
async def summarise(req: SummariseRequest:
    user_text = req.text

@app.post("/tool-call")
async def tool_call_stub(req: ChatRequest):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    resp = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=req.messages,
        tools=tools,
        tool_choice="auto",
    )
    return resp.choices[0].message.model_dump()
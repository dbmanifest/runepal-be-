from .buddie import buddie_agent
from fastapi import FastAPI
from openai import OpenAI
from typing import List, Union, Dict, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

import uvicorn
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GPT_MODEL = "gpt-3.5-turbo-0125"
EMBEDDING_MODEL = "text-embedding-ada-002"


class LogitBiasItem(BaseModel):
    token: str
    bias: int

class Function(BaseModel):
    name: str

class ToolChoice(BaseModel):
    type: str
    function: Optional[Function]
class ResponseFormat(BaseModel):
    type: str

class Stop(BaseModel):
    sequence: Union[str, List[str]]

class RequestBody(BaseModel):
    messages: List[Dict[str,str]]
    model: str
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Stop] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1
    tools: Optional[List[str]] = None
    tool_choice: Optional[ToolChoice] = None


@app.post("/v1/chat/completions/")
async def ask(request: RequestBody):
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    messages = request.messages
    query = messages.pop()
    result = buddie_agent(query,messages)
    response = {
        "id": "civicpal-123",
        "object": "civic.completion",
        "created": 1677652288,
        "model": "mechpal",
        "system_fingerprint": "0",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result,
            },
            "logprobs": None,
            "finish_reason": "stop"
        }],
            "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }
    response = jsonable_encoder(response)
    print(response)
    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=True)
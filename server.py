import sys
import asyncio
from fastapi import FastAPI
from typing import Optional, List
from pydantic import BaseModel, Field

app = FastAPI(title="OpenAPI Assistant", contact={
    "name": "Assistant",
    "email": "support@openai.com"}, description="A REST API for interacting with OpenAI models.")


class TaskRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0
    stop: Optional[List[str]] = None
    stream: bool = False
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    message_counts: Optional[int] = None


def render_markdown(text: str) -> str:
    result = text
    block1 = result.split("\n\n")
    if len(block1) > 1:
        result = "\n".join([f"{i+1}. {line}" if i < len(block1)-2 else line for i, line in enumerate(block1)])
    return result


async def generate_response(task: TaskRequest) -> dict:
    try:
        # openai >= 1.0 style call
        completion = await client.chat.completions.create(
            messages=[{"role": "system", "content": task.system or ""}] + [
                {"role": "user", "content": task.prompt}
            ],
            model="gpt-4o",
        )
        return {"id": completion.id, "created": int(time.strftime("%Y%m%dT%H%M%SZ")),
                "object": "assistant_message",
                "choices": [{"index": 0, "message": {
                    "role": "assistant",
                    "content": render_markdown(completion.choices[0].message.content)},
                 }]}],
                "usage":{"total_tokens": int(completion.usage.total_tokens)}}
    except Exception as e:
        return {"error": str(e)}


client = AsyncClient()


async def start():
    await client.wait_for_completion()
    print("OpenAI server started on http://localhost:8000")
    print("Browse documentation at http://localhost:8000/docs")
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(start())

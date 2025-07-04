import os
from agents import Agent, Runner, AsyncOpenAI,set_tracing_disabled, OpenAIChatCompletionsModel, RunConfig # type: ignore
from dotenv import load_dotenv
import chainlit as cl # type: ignore
from openai.types.responses import ResponseTextDeltaEvent # type: ignore

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

agent: Agent = Agent(name="Assistant", instructions="Your are a helpful assistant", model=model)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="made by kavita luhana").send()



@cl.on_message
async def handle_message(message: cl.Message):

    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()
    
    history.append({"role": "user", "content": message.content})


    result = Runner.run_streamed(
        agent,
        input=history
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
    history.append({"role": "assistant", "content": result.final_output})


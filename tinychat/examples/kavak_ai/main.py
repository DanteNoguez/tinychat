import asyncio
import os

from dotenv import load_dotenv

# isort: off
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware import Middleware

load_dotenv()

from tinychat.examples.kavak_ai.agents import KAVAK_AGENTS_REGISTRY, AgentConfigType

# from tinychat.examples.kavak_ai.utils.rag import KAVAK_VECTOR_REGISTRY
from tinychat.examples.kavak_ai.chat import KavakConversationsManager
from tinychat.examples.kavak_ai.models import ChatRequest, ChatResponse, MemoryRequest

# isort: on
from tinychat.logging import configure_pretty_logging

configure_pretty_logging()

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ),
]

app = FastAPI(docs_url="/docs")

CONVERSATIONS_MANAGER = KavakConversationsManager(
    agent_registry=KAVAK_AGENTS_REGISTRY,
    # vectordb_registry=KAVAK_VECTOR_REGISTRY,
)


@app.post("/chat/send")
async def chat(request: ChatRequest) -> ChatResponse:
    response = await CONVERSATIONS_MANAGER.handle_incoming_message(
        identifier=request.session_id,
        request_type=AgentConfigType.DEFAULT,
        message=request.message,
    )
    return ChatResponse(content=response)


@app.get("/chat/get")
async def get_chat(request: MemoryRequest) -> ChatResponse:
    response = await CONVERSATIONS_MANAGER.get_conversation(
        identifier=request.session_id,
    )
    return ChatResponse(content=response)


@app.delete("/chat/remove")
async def get_chat(request: MemoryRequest) -> ChatResponse:
    response = await CONVERSATIONS_MANAGER.remove_conversation(
        identifier=request.session_id,
    )
    return ChatResponse(content=response)


@app.post("/twilio/webhook")
async def twilio_webhook(request: Request) -> Response:
    form_data = await request.form()
    from_phone = form_data.get("From")
    to_phone = form_data.get("To")
    body = form_data.get("Body")
    logger.info(f"Received webhook: to={to_phone}, from={from_phone}, body={body}")
    asyncio.create_task(
        CONVERSATIONS_MANAGER.handle_incoming_whatsapp(
            our_phone=to_phone, their_phone=from_phone, message=body
        )
    )
    return Response(content="success", status_code=200)


async def startup_event():
    logger.debug("starting...")


app.add_event_handler("startup", startup_event)

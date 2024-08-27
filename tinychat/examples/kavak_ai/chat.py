import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)

from loguru import logger
from twilio.rest import Client as TwilioClient

from tinychat.examples.kavak_ai.agents import AgentConfigType
from tinychat.manager.in_memory import InMemoryConversationsManager


class KavakConversationsManager(InMemoryConversationsManager):
    def __init__(
        self,
        twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID"),
        twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = TwilioClient(twilio_account_sid, twilio_auth_token)

    async def handle_incoming_whatsapp(
        self, our_phone: str, their_phone: str, message: str
    ) -> None:
        conversation = self._get_or_create_conversation(
            self.get_conversation_id(our_phone + their_phone),
            request_type=AgentConfigType.DEFAULT,
        )
        response = await conversation.generate_response(message)
        self.send_whatsapp(our_phone, their_phone, response)

    def send_whatsapp(self, our_phone: str, their_phone: str, message: str) -> None:
        try:
            response = self.client.messages.create(
                from_=our_phone,
                body=message,
                to=their_phone,
            )
            logger.debug(
                f"WhatsApp message: {message}\nSent to: {their_phone}\nStatus: {response.status}"
            )
        except Exception as e:
            logger.error(f"Error sending WhatsApp to {their_phone}: {e}", exc_info=True)
            raise e

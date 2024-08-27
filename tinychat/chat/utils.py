import secrets


def create_conversation_id():
    return secrets.token_urlsafe(16)

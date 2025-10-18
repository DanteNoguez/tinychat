import secrets


def random_id() -> str:
    return secrets.token_hex(2)

from enum import Enum

class LLMServiceType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
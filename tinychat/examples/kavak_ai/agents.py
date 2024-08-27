from tinychat.agents import OpenAIAgent
from tinychat.agents.models import OpenAIAgentConfig
from tinychat.agents.registry import AgentConfigRegistry
from tinychat.examples.kavak_ai.models import AgentConfigType
from tinychat.examples.kavak_ai.prompts import DEFAULT_KAVAK_AGENT_PROMPT

# isort: off
from tinychat.examples.kavak_ai.tools import (
    BUSCAR_AUTO_TOOL,
    FINANCIAMIENTO_TOOL,
    KAVAK_PANDAS_AGENT_CONFIG,
)

# isort: on

KAVAK_DEFAULT_AGENT_CONFIG = OpenAIAgentConfig(
    prompt=DEFAULT_KAVAK_AGENT_PROMPT, tools=[BUSCAR_AUTO_TOOL, FINANCIAMIENTO_TOOL]
)

KAVAK_DEFAULT_AGENT = OpenAIAgent(config=KAVAK_DEFAULT_AGENT_CONFIG)

KAVAK_AGENTS_REGISTRY = AgentConfigRegistry(
    configs={
        AgentConfigType.DEFAULT: KAVAK_DEFAULT_AGENT_CONFIG,
        AgentConfigType.PANDAS: KAVAK_PANDAS_AGENT_CONFIG,
    },
    default_agent_type=AgentConfigType.DEFAULT,
)

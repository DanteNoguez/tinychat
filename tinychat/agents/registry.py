from tinychat.agents.models import AgentConfig
from tinychat.agents.openai_agent import OpenAIAgent


class AgentConfigRegistry:
    def __init__(self, configs: dict[str, AgentConfig], default_agent_type: str):
        self.configs = configs
        self.default_agent_type = default_agent_type

    def register_config(self, agent_type: str, config: AgentConfig):
        self.configs[agent_type] = config

    def set_default_agent_type(self, agent_type: str):
        if agent_type not in self.configs:
            raise ValueError(f"Agent config '{agent_type}' not found in registry")
        self.default_agent_type = agent_type

    def get_config(self, agent_type: str) -> AgentConfig:
        return self.configs.get(agent_type, self.configs[self.default_agent_type])

    def get_agent(self, agent_type: str) -> OpenAIAgent:
        config = self.get_config(agent_type)
        return OpenAIAgent(config)

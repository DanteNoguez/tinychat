"""Tests for OpenAI agent (basic instantiation only)."""


from tinychat.services.openai.llm.models import OpenAIAgentConfig, Tool, ToolParameter
from tinychat.services.openai.llm.openai_agent import OpenAIAgent


class TestOpenAIAgent:
    """Test OpenAIAgent class (basic instantiation)."""

    def test_openai_agent_instantiation(self):
        """Test that OpenAIAgent can be instantiated."""
        config = OpenAIAgentConfig(
            prompt="You are a helpful assistant",
            api_key="test-key",  # Use fake key for instantiation test
        )
        agent = OpenAIAgent(config)

        assert agent is not None
        assert agent.model_name == "gpt-4o"
        assert agent.temperature == 0.2
        assert agent.max_tokens == 300

    def test_openai_agent_with_custom_config(self):
        """Test OpenAIAgent with custom configuration."""
        config = OpenAIAgentConfig(
            prompt="Custom prompt",
            model_name="gpt-3.5-turbo",
            temperature=0.8,
            max_tokens=200,
            api_key="test-key",
            max_retries=5,
        )
        agent = OpenAIAgent(config)

        assert agent.model_name == "gpt-3.5-turbo"
        assert agent.temperature == 0.8
        assert agent.max_tokens == 200

    def test_openai_agent_generate_prompt(self):
        """Test prompt generation."""
        config = OpenAIAgentConfig(
            prompt="You are a helpful assistant", api_key="test-key"
        )
        agent = OpenAIAgent(config)

        prompt = agent.generate_prompt("Test prompt")
        assert len(prompt) == 1
        assert prompt[0]["role"] == "system"
        assert prompt[0]["content"] == "Test prompt"

    def test_openai_agent_without_tools(self):
        """Test agent without tools."""
        config = OpenAIAgentConfig(prompt="Test", api_key="test-key")
        agent = OpenAIAgent(config)

        assert agent.tools is None
        assert agent.tools_schema is None

    def test_openai_agent_with_tools(self):
        """Test agent with tools."""
        param = ToolParameter(
            name="city", description="The city name", data_type="string"
        )
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters=[param],
        )

        config = OpenAIAgentConfig(prompt="Test", api_key="test-key", tools=[tool])
        agent = OpenAIAgent(config)

        assert len(agent.tools) == 1
        assert agent.tools_schema is not None
        assert len(agent.tools_schema) == 1
        assert agent.tools_schema[0]["function"]["name"] == "get_weather"

    def test_openai_agent_create_tools_schema(self):
        """Test tools schema creation."""
        param1 = ToolParameter(name="city", description="City name", data_type="string")
        param2 = ToolParameter(name="units", description="Units", data_type="string")

        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters=[param1, param2],
        )

        config = OpenAIAgentConfig(prompt="Test", api_key="test-key", tools=[tool])
        agent = OpenAIAgent(config)
        schema = agent.create_tools_schema([tool])

        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "get_weather"
        assert schema[0]["function"]["strict"] is True
        assert "city" in schema[0]["function"]["parameters"]["properties"]
        assert "units" in schema[0]["function"]["parameters"]["properties"]
        assert len(schema[0]["function"]["parameters"]["required"]) == 2

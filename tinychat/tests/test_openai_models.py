"""Tests for OpenAI models."""
import pytest

from tinychat.services.openai.llm.models import (
    ToolParameter,
    Tool,
    AgentConfig,
    OpenAIAgentConfig,
)


class TestToolParameter:
    """Test ToolParameter class."""

    def test_tool_parameter_instantiation(self):
        """Test that ToolParameter can be instantiated."""
        param = ToolParameter(
            name="city", description="The city name", data_type="string"
        )
        assert param.name == "city"
        assert param.description == "The city name"
        assert param.data_type == "string"


class TestTool:
    """Test Tool class."""

    def test_tool_instantiation(self):
        """Test that Tool can be instantiated."""
        param = ToolParameter(
            name="city", description="The city name", data_type="string"
        )
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters=[param],
        )
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a city"
        assert len(tool.parameters) == 1

    @pytest.mark.asyncio
    async def test_tool_run_not_implemented(self):
        """Test that Tool.run raises NotImplementedError."""
        tool = Tool(name="test", description="test", parameters=[])
        
        with pytest.raises(NotImplementedError):
            await tool.run()


class TestAgentConfig:
    """Test AgentConfig class."""

    def test_agent_config_instantiation(self):
        """Test that AgentConfig can be instantiated."""
        config = AgentConfig(
            prompt="You are helpful",
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=100,
        )
        assert config.prompt == "You are helpful"
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 100
        assert config.api_key is None
        assert config.max_retries is None
        assert config.tools is None

    def test_agent_config_with_optional_params(self):
        """Test AgentConfig with optional parameters."""
        param = ToolParameter(name="x", description="test", data_type="string")
        tool = Tool(name="test", description="test", parameters=[param])
        
        config = AgentConfig(
            prompt="You are helpful",
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=100,
            api_key="test-key",
            max_retries=3,
            tools=[tool],
        )
        assert config.api_key == "test-key"
        assert config.max_retries == 3
        assert len(config.tools) == 1


class TestOpenAIAgentConfig:
    """Test OpenAIAgentConfig class."""

    def test_openai_agent_config_defaults(self):
        """Test OpenAIAgentConfig with defaults."""
        # Skip if OPENAI_API_KEY is not set
        import os
        if "OPENAI_API_KEY" not in os.environ:
            pytest.skip("OPENAI_API_KEY not set")

        config = OpenAIAgentConfig(prompt="You are helpful")
        assert config.prompt == "You are helpful"
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.2
        assert config.max_tokens == 300
        assert config.max_retries == 2

    def test_openai_agent_config_custom_values(self):
        """Test OpenAIAgentConfig with custom values."""
        config = OpenAIAgentConfig(
            prompt="You are helpful",
            model_name="gpt-3.5-turbo",
            temperature=0.9,
            max_tokens=500,
            api_key="test-key",
            max_retries=5,
        )
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.9
        assert config.max_tokens == 500
        assert config.api_key == "test-key"
        assert config.max_retries == 5


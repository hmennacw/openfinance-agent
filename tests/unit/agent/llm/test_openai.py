import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import os

from src.agent.llm.openai import OpenAIProvider

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = AsyncMock()
    
    # Mock chat completions
    mock_completion = AsyncMock()
    mock_completion.choices = [AsyncMock(message=AsyncMock(content="mock response"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    # Mock threads
    mock_thread = AsyncMock(id="mock_thread_id")
    mock_client.beta.threads.create = AsyncMock(return_value=mock_thread)
    
    # Mock messages
    mock_message = AsyncMock(
        id="mock_message_id",
        role="assistant",
        content=[AsyncMock(text=AsyncMock(value="mock response"))],
        created_at="2024-01-01T00:00:00Z"
    )
    mock_client.beta.threads.messages.create = AsyncMock(return_value=mock_message)
    mock_client.beta.threads.messages.list = AsyncMock(return_value=AsyncMock(data=[mock_message]))
    
    # Mock runs
    mock_run = AsyncMock(id="mock_run_id", status="completed")
    mock_client.beta.threads.runs.create = AsyncMock(return_value=mock_run)
    mock_client.beta.threads.runs.retrieve = AsyncMock(return_value=mock_run)
    
    # Mock API key validation
    mock_client.api_key = "test_key"
    
    return mock_client

class TestOpenAIProvider:
    """Test suite for OpenAIProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create an OpenAI provider instance."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "mock_api_key"}):
            return OpenAIProvider()
    
    def test_initialization_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = OpenAIProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert provider.model == "gpt-4-1106-preview"  # default model
    
    def test_initialization_with_env_var(self):
        """Test initialization with API key from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"}):
            provider = OpenAIProvider()
            assert provider.api_key == "env_key"
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAIProvider()
            assert "OpenAI API key must be provided" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_initialize_client(self, provider, mock_openai_client):
        """Test client initialization."""
        with patch("openai.AsyncOpenAI", return_value=mock_openai_client):
            await provider.initialize()
            assert provider.client is not None
    
    @pytest.mark.asyncio
    async def test_generate_completion(self, provider, mock_openai_client):
        """Test completion generation."""
        provider.client = mock_openai_client
        
        completion = await provider.generate_completion(
            prompt="test prompt",
            context={"system": "test context"},
            temperature=0.5,
            max_tokens=100
        )
        
        assert completion == "mock response"
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=provider.model,
            messages=[
                {"role": "system", "content": "{'system': 'test context'}"},
                {"role": "user", "content": "test prompt"}
            ],
            temperature=0.5,
            max_tokens=100
        )
    
    @pytest.mark.asyncio
    async def test_generate_completion_without_context(self, provider, mock_openai_client):
        """Test completion generation without context."""
        provider.client = mock_openai_client
        
        await provider.generate_completion("test prompt")
        
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=provider.model,
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.7,
            max_tokens=None
        )
    
    @pytest.mark.asyncio
    async def test_create_thread(self, provider, mock_openai_client):
        """Test thread creation."""
        provider.client = mock_openai_client
        
        thread_id = await provider.create_thread()
        
        assert thread_id == "mock_thread_id"
        mock_openai_client.beta.threads.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_message_to_thread(self, provider, mock_openai_client):
        """Test adding message to thread."""
        provider.client = mock_openai_client
        
        await provider.add_message_to_thread(
            thread_id="test_thread",
            content="test message",
            role="user"
        )
        
        mock_openai_client.beta.threads.messages.create.assert_called_once_with(
            thread_id="test_thread",
            role="user",
            content="test message"
        )
    
    @pytest.mark.asyncio
    async def test_run_thread_success(self, provider, mock_openai_client):
        """Test successful thread run."""
        provider.client = mock_openai_client
        
        response = await provider.run_thread(
            thread_id="test_thread",
            assistant_id="test_assistant",
            instructions="test instructions"
        )
        
        assert response == "mock response"
        mock_openai_client.beta.threads.runs.create.assert_called_once_with(
            thread_id="test_thread",
            assistant_id="test_assistant",
            instructions="test instructions"
        )
    
    @pytest.mark.asyncio
    async def test_run_thread_failure(self, provider, mock_openai_client):
        """Test thread run failure."""
        provider.client = mock_openai_client
        
        # Mock run status to be failed
        failed_run = AsyncMock(id="mock_run_id", status="failed")
        mock_openai_client.beta.threads.runs.retrieve = AsyncMock(return_value=failed_run)
        
        with pytest.raises(Exception) as exc_info:
            await provider.run_thread(
                thread_id="test_thread",
                assistant_id="test_assistant"
            )
        assert "Run failed with status: failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_thread_messages(self, provider, mock_openai_client):
        """Test getting thread messages."""
        provider.client = mock_openai_client
        
        messages = await provider.get_thread_messages(
            thread_id="test_thread",
            limit=10
        )
        
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "mock response"
        assert messages[0]["created_at"] == "2024-01-01T00:00:00Z"
        
        mock_openai_client.beta.threads.messages.list.assert_called_once_with(
            thread_id="test_thread",
            limit=10
        )
    
    @pytest.mark.asyncio
    async def test_auto_initialization(self, provider, mock_openai_client):
        """Test automatic client initialization."""
        with patch.object(provider, 'initialize') as mock_initialize:
            # Configure the mock to return None
            mock_initialize.return_value = None
            
            # Call a method without initializing first
            try:
                await provider.generate_completion("test")
            except:
                pass  # We expect an error since we're mocking initialize
            
            # Verify initialize was called
            mock_initialize.assert_called_once() 
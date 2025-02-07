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
        # Ensure client is not initialized
        provider.client = None
        
        # Mock the initialize method and configure it to set the client
        with patch.object(provider, 'initialize') as mock_initialize:
            mock_initialize.side_effect = lambda: setattr(provider, 'client', mock_openai_client)
            
            # Configure mock response
            mock_completion = AsyncMock()
            mock_completion.choices = [AsyncMock(message=AsyncMock(content="mock response"))]
            mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            
            # Call method that should trigger initialization
            response = await provider.generate_completion("test")
            
            # Verify initialize was called and response is correct
            mock_initialize.assert_called_once()
            assert response == "mock response"
            
    @pytest.mark.asyncio
    async def test_run_thread_cancelled(self, provider, mock_openai_client):
        """Test thread run cancelled."""
        provider.client = mock_openai_client
        
        # Mock run status to be cancelled
        cancelled_run = AsyncMock(id="mock_run_id", status="cancelled")
        mock_openai_client.beta.threads.runs.retrieve = AsyncMock(return_value=cancelled_run)
        
        with pytest.raises(Exception) as exc_info:
            await provider.run_thread(
                thread_id="test_thread",
                assistant_id="test_assistant"
            )
        assert "Run failed with status: cancelled" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_run_thread_expired(self, provider, mock_openai_client):
        """Test thread run expired."""
        provider.client = mock_openai_client
        
        # Mock run status to be expired
        expired_run = AsyncMock(id="mock_run_id", status="expired")
        mock_openai_client.beta.threads.runs.retrieve = AsyncMock(return_value=expired_run)
        
        with pytest.raises(Exception) as exc_info:
            await provider.run_thread(
                thread_id="test_thread",
                assistant_id="test_assistant"
            )
        assert "Run failed with status: expired" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_thread_messages_auto_init(self, provider, mock_openai_client):
        """Test get thread messages with auto initialization."""
        # Ensure client is not initialized
        provider.client = None
        
        # Mock the initialize method and configure it to set the client
        with patch.object(provider, 'initialize') as mock_initialize:
            mock_initialize.side_effect = lambda: setattr(provider, 'client', mock_openai_client)
            
            # Configure mock response
            mock_messages = [
                AsyncMock(
                    role="user",
                    content=[AsyncMock(text=AsyncMock(value="Hello"))],
                    created_at="2024-01-01T00:00:00Z"
                ),
                AsyncMock(
                    role="assistant",
                    content=[AsyncMock(text=AsyncMock(value="Hi there"))],
                    created_at="2024-01-01T00:00:01Z"
                )
            ]
            mock_response = AsyncMock(data=mock_messages)
            mock_openai_client.beta.threads.messages.list = AsyncMock(return_value=mock_response)
            
            # Call method that should trigger initialization
            messages = await provider.get_thread_messages("test_thread")
            
            # Verify initialize was called and response is correct
            mock_initialize.assert_called_once()
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "Hi there"
    
    @pytest.mark.asyncio
    async def test_initialize_with_env_var_no_api_key(self):
        """Test initialization with environment variable but no API key."""
        # First create with a temporary key
        provider = OpenAIProvider(api_key="temp_key")
        
        # Then remove the key and try to initialize without environment variable
        provider.api_key = None
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                await provider.initialize()
            assert str(exc_info.value) == "No API key provided"
        
        # Now set the environment variable and try again
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            await provider.initialize()
            assert provider.api_key == "test_key"
            assert provider.client is not None
    
    @pytest.mark.asyncio
    async def test_run_thread_cancelled_status(self, provider, mock_openai_client):
        """Test run thread with cancelled status."""
        # Ensure client is not initialized
        provider.client = None
        
        # Mock the initialize method
        async def mock_initialize():
            provider.client = mock_openai_client
        provider.initialize = mock_initialize
        
        # Configure mock response for cancelled status
        mock_run = AsyncMock(id="run_id", status="cancelled")
        mock_openai_client.beta.threads.runs.create = AsyncMock(return_value=AsyncMock(id="run_id"))
        mock_openai_client.beta.threads.runs.retrieve = AsyncMock(return_value=mock_run)
        
        with pytest.raises(Exception) as exc_info:
            await provider.run_thread("thread_id", "assistant_id")
        assert str(exc_info.value) == "Run failed with status: cancelled"
        
        # Verify the retrieve call was made
        mock_openai_client.beta.threads.runs.retrieve.assert_called_with(
            thread_id="thread_id",
            run_id="run_id"
        )
    
    @pytest.mark.asyncio
    async def test_run_thread_expired_status(self, provider, mock_openai_client):
        """Test run thread with expired status."""
        # Ensure client is not initialized
        provider.client = None
        
        # Mock the initialize method
        with patch.object(provider, 'initialize') as mock_initialize:
            mock_initialize.side_effect = lambda: setattr(provider, 'client', mock_openai_client)
            
            # Configure mock response for expired status
            mock_run = AsyncMock(id="run_id", status="expired")
            mock_openai_client.beta.threads.runs.create = AsyncMock(return_value=AsyncMock(id="run_id"))
            mock_openai_client.beta.threads.runs.retrieve = AsyncMock(return_value=mock_run)
            
            with pytest.raises(Exception) as exc_info:
                await provider.run_thread("thread_id", "assistant_id")
            assert str(exc_info.value) == "Run failed with status: expired"
            
            # Verify initialize was called
            mock_initialize.assert_called_once()
            
            # Verify the retrieve call was made
            mock_openai_client.beta.threads.runs.retrieve.assert_called_with(
                thread_id="thread_id",
                run_id="run_id"
            )
    
    @pytest.mark.asyncio
    async def test_run_thread_auto_init(self, provider, mock_openai_client):
        """Test run thread with auto initialization."""
        # Ensure client is not initialized
        provider.client = None
        
        # Configure mock response
        mock_run = AsyncMock(id="run_id", status="completed")
        mock_openai_client.beta.threads.runs.create = AsyncMock(return_value=AsyncMock(id="run_id"))
        mock_openai_client.beta.threads.runs.retrieve = AsyncMock(return_value=mock_run)
        mock_messages = AsyncMock(data=[
            AsyncMock(
                content=[AsyncMock(text=AsyncMock(value="mock response"))]
            )
        ])
        mock_openai_client.beta.threads.messages.list = AsyncMock(return_value=mock_messages)
        
        # Mock the initialize method
        with patch.object(provider, 'initialize', new_callable=AsyncMock) as mock_initialize:
            mock_initialize.side_effect = lambda: setattr(provider, 'client', mock_openai_client)
            
            # Call run_thread
            response = await provider.run_thread("thread_id", "assistant_id")
            assert response == "mock response"
            
            # Verify initialize was called
            mock_initialize.assert_called_once()
            
            # Verify the calls were made in the correct order
            mock_openai_client.beta.threads.runs.create.assert_called_once()
            mock_openai_client.beta.threads.runs.retrieve.assert_called_once()
            mock_openai_client.beta.threads.messages.list.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_thread_with_status_transitions(self, provider, mock_openai_client):
        """Test run thread with different status transitions."""
        # Ensure client is not initialized
        provider.client = None
        
        # Mock the initialize method
        with patch.object(provider, 'initialize', new_callable=AsyncMock) as mock_initialize:
            mock_initialize.side_effect = lambda: setattr(provider, 'client', mock_openai_client)
            
            # Configure mock responses for status transitions
            in_progress_run = AsyncMock(id="run_id", status="in_progress")
            completed_run = AsyncMock(id="run_id", status="completed")
            
            mock_openai_client.beta.threads.runs.create = AsyncMock(return_value=AsyncMock(id="run_id"))
            mock_openai_client.beta.threads.runs.retrieve.side_effect = [
                in_progress_run,  # First call returns in_progress
                completed_run     # Second call returns completed
            ]
            
            mock_messages = AsyncMock(data=[
                AsyncMock(
                    content=[AsyncMock(text=AsyncMock(value="mock response"))]
                )
            ])
            mock_openai_client.beta.threads.messages.list = AsyncMock(return_value=mock_messages)
            
            # Run the thread
            response = await provider.run_thread("thread_id", "assistant_id")
            
            # Verify initialize was called
            mock_initialize.assert_called_once()
            
            # Verify the calls were made in the correct order
            mock_openai_client.beta.threads.runs.create.assert_called_once_with(
                thread_id="thread_id",
                assistant_id="assistant_id",
                instructions=None
            )
            
            # Verify retrieve was called twice with different statuses
            assert mock_openai_client.beta.threads.runs.retrieve.call_count == 2
            mock_openai_client.beta.threads.runs.retrieve.assert_called_with(
                thread_id="thread_id",
                run_id="run_id"
            )
            
            # Verify the final response
            assert response == "mock response"
    
    @pytest.mark.asyncio
    async def test_run_thread_queued_status(self, provider, mock_openai_client):
        """Test run thread with queued status transition."""
        # Ensure client is not initialized
        provider.client = None
        
        # Mock the initialize method
        with patch.object(provider, 'initialize', new_callable=AsyncMock) as mock_initialize:
            mock_initialize.side_effect = lambda: setattr(provider, 'client', mock_openai_client)
            
            # Configure mock responses for status transitions
            queued_run = AsyncMock(id="run_id", status="queued")
            in_progress_run = AsyncMock(id="run_id", status="in_progress")
            completed_run = AsyncMock(id="run_id", status="completed")
            
            mock_openai_client.beta.threads.runs.create = AsyncMock(return_value=AsyncMock(id="run_id"))
            mock_openai_client.beta.threads.runs.retrieve.side_effect = [
                queued_run,      # First call returns queued
                in_progress_run, # Second call returns in_progress
                completed_run    # Third call returns completed
            ]
            
            mock_messages = AsyncMock(data=[
                AsyncMock(
                    content=[AsyncMock(text=AsyncMock(value="mock response"))]
                )
            ])
            mock_openai_client.beta.threads.messages.list = AsyncMock(return_value=mock_messages)
            
            # Run the thread
            response = await provider.run_thread("thread_id", "assistant_id")
            
            # Verify initialize was called
            mock_initialize.assert_called_once()
            
            # Verify the calls were made in the correct order
            mock_openai_client.beta.threads.runs.create.assert_called_once()
            
            # Verify retrieve was called three times with different statuses
            assert mock_openai_client.beta.threads.runs.retrieve.call_count == 3
            
            # Verify the final response
            assert response == "mock response"
    
    @pytest.mark.asyncio
    async def test_thread_operations_auto_init(self, provider, mock_openai_client):
        """Test thread operations with auto initialization."""
        # Ensure client is not initialized
        provider.client = None
        
        # Mock the initialize method
        with patch.object(provider, 'initialize', new_callable=AsyncMock) as mock_initialize:
            mock_initialize.side_effect = lambda: setattr(provider, 'client', mock_openai_client)
            
            # Test create_thread
            mock_thread = AsyncMock(id="thread_id")
            mock_openai_client.beta.threads.create = AsyncMock(return_value=mock_thread)
            
            thread_id = await provider.create_thread()
            assert thread_id == "thread_id"
            mock_initialize.assert_called_once()
            mock_openai_client.beta.threads.create.assert_called_once()
            
            # Reset initialize mock for next test
            mock_initialize.reset_mock()
            provider.client = None
            
            # Test add_message_to_thread
            mock_message = AsyncMock(id="message_id")
            mock_openai_client.beta.threads.messages.create = AsyncMock(return_value=mock_message)
            
            await provider.add_message_to_thread("thread_id", "test message", "user")
            mock_initialize.assert_called_once()
            mock_openai_client.beta.threads.messages.create.assert_called_once_with(
                thread_id="thread_id",
                role="user",
                content="test message"
            ) 
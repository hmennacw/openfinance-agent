import pytest
from typing import Dict, Any, List, Optional
from src.agent.llm.base import LLMProvider

class MockLLMProvider(LLMProvider):
    """Mock implementation of LLMProvider for testing."""
    
    def __init__(self):
        self.initialized = False
        self.threads: Dict[str, List[Dict[str, Any]]] = {}
        self.completion_response = "mock completion"
        self.thread_response = "mock thread response"
    
    async def initialize(self) -> None:
        """Initialize the mock provider."""
        self.initialized = True
    
    async def generate_completion(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a mock completion."""
        return self.completion_response
    
    async def create_thread(self) -> str:
        """Create a mock thread."""
        thread_id = f"thread_{len(self.threads)}"
        self.threads[thread_id] = []
        return thread_id
    
    async def add_message_to_thread(
        self,
        thread_id: str,
        content: str,
        role: str = "user"
    ) -> None:
        """Add a message to the mock thread."""
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} not found")
        
        self.threads[thread_id].append({
            "role": role,
            "content": content
        })
    
    async def run_thread(
        self,
        thread_id: str,
        assistant_id: str,
        instructions: Optional[str] = None
    ) -> str:
        """Run the mock thread."""
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} not found")
        return self.thread_response
    
    async def get_thread_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages from the mock thread."""
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} not found")
        
        messages = self.threads[thread_id]
        if limit is not None:
            messages = messages[-limit:]
        return messages

class TestLLMProvider:
    """Test suite for LLMProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create a mock LLM provider instance."""
        return MockLLMProvider()
    
    @pytest.mark.asyncio
    async def test_initialization(self, provider):
        """Test provider initialization."""
        assert not provider.initialized
        await provider.initialize()
        assert provider.initialized
    
    @pytest.mark.asyncio
    async def test_generate_completion(self, provider):
        """Test completion generation."""
        completion = await provider.generate_completion(
            prompt="test prompt",
            context={"test": "context"},
            temperature=0.5,
            max_tokens=100
        )
        assert completion == provider.completion_response
    
    @pytest.mark.asyncio
    async def test_thread_creation(self, provider):
        """Test thread creation."""
        thread_id = await provider.create_thread()
        assert thread_id.startswith("thread_")
        assert thread_id in provider.threads
        assert provider.threads[thread_id] == []
    
    @pytest.mark.asyncio
    async def test_add_message_to_thread(self, provider):
        """Test adding messages to a thread."""
        thread_id = await provider.create_thread()
        await provider.add_message_to_thread(
            thread_id=thread_id,
            content="test message",
            role="user"
        )
        
        messages = provider.threads[thread_id]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test message"
    
    @pytest.mark.asyncio
    async def test_add_message_to_nonexistent_thread(self, provider):
        """Test adding message to nonexistent thread."""
        with pytest.raises(ValueError) as exc_info:
            await provider.add_message_to_thread(
                thread_id="nonexistent",
                content="test message"
            )
        assert "Thread nonexistent not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_run_thread(self, provider):
        """Test running a thread."""
        thread_id = await provider.create_thread()
        await provider.add_message_to_thread(thread_id, "test message")
        
        response = await provider.run_thread(
            thread_id=thread_id,
            assistant_id="test_assistant",
            instructions="test instructions"
        )
        assert response == provider.thread_response
    
    @pytest.mark.asyncio
    async def test_run_nonexistent_thread(self, provider):
        """Test running nonexistent thread."""
        with pytest.raises(ValueError) as exc_info:
            await provider.run_thread(
                thread_id="nonexistent",
                assistant_id="test_assistant"
            )
        assert "Thread nonexistent not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_thread_messages(self, provider):
        """Test getting thread messages."""
        thread_id = await provider.create_thread()
        
        # Add multiple messages
        messages = [
            ("Hello", "user"),
            ("Hi there!", "assistant"),
            ("How are you?", "user")
        ]
        
        for content, role in messages:
            await provider.add_message_to_thread(thread_id, content, role)
        
        # Test getting all messages
        all_messages = await provider.get_thread_messages(thread_id)
        assert len(all_messages) == len(messages)
        
        # Test getting limited messages
        limited_messages = await provider.get_thread_messages(thread_id, limit=2)
        assert len(limited_messages) == 2
        assert limited_messages[-1]["content"] == "How are you?"
    
    @pytest.mark.asyncio
    async def test_get_messages_from_nonexistent_thread(self, provider):
        """Test getting messages from nonexistent thread."""
        with pytest.raises(ValueError) as exc_info:
            await provider.get_thread_messages("nonexistent")
        assert "Thread nonexistent not found" in str(exc_info.value) 
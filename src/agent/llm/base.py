from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM provider with necessary setup."""
        pass
    
    @abstractmethod
    async def generate_completion(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        pass
    
    @abstractmethod
    async def create_thread(self) -> str:
        """Create a new conversation thread."""
        pass
    
    @abstractmethod
    async def add_message_to_thread(
        self,
        thread_id: str,
        content: str,
        role: str = "user"
    ) -> None:
        """Add a message to an existing thread."""
        pass
    
    @abstractmethod
    async def run_thread(
        self,
        thread_id: str,
        assistant_id: str,
        instructions: Optional[str] = None
    ) -> str:
        """Run the assistant on the thread and get the response."""
        pass
    
    @abstractmethod
    async def get_thread_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages from a thread."""
        pass 
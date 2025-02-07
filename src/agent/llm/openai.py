from typing import Any, Dict, List, Optional
import os

from openai import AsyncOpenAI

from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    """OpenAI implementation of the LLM provider interface."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-1106-preview"
    ):
        """Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If not provided, will try to get from environment.
            model: The model to use for completions.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.model = model
        self.client: Optional[AsyncOpenAI] = None
        
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate_completion(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a completion using the OpenAI API."""
        if not self.client:
            await self.initialize()
            
        messages = [{"role": "user", "content": prompt}]
        if context:
            messages.insert(0, {"role": "system", "content": str(context)})
            
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def create_thread(self) -> str:
        """Create a new thread using the OpenAI API."""
        if not self.client:
            await self.initialize()
            
        thread = await self.client.beta.threads.create()
        return thread.id
    
    async def add_message_to_thread(
        self,
        thread_id: str,
        content: str,
        role: str = "user"
    ) -> None:
        """Add a message to an existing thread."""
        if not self.client:
            await self.initialize()
            
        await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )
    
    async def run_thread(
        self,
        thread_id: str,
        assistant_id: str,
        instructions: Optional[str] = None
    ) -> str:
        """Run the assistant on the thread and get the response."""
        if not self.client:
            await self.initialize()
            
        run = await self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions=instructions
        )
        
        # Wait for the run to complete
        while True:
            run_status = await self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Run failed with status: {run_status.status}")
                
        # Get the latest message
        messages = await self.client.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content[0].text.value
    
    async def get_thread_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages from a thread."""
        if not self.client:
            await self.initialize()
            
        messages = await self.client.beta.threads.messages.list(
            thread_id=thread_id,
            limit=limit
        )
        
        return [
            {
                "role": msg.role,
                "content": msg.content[0].text.value,
                "created_at": msg.created_at
            }
            for msg in messages.data
        ] 
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

@dataclass
class Memory:
    """Base class for memory items."""
    content: Any
    memory_type: str = "generic"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeMemory:
    """Memory specific to code generation."""
    content: Any
    file_path: str
    code_type: str  # e.g., "handler", "usecase", "model"
    dependencies: List[str] = field(default_factory=list)
    _base: Memory = field(init=False)
    
    def __post_init__(self):
        self._base = Memory(
            content=self.content,
            memory_type="code",
            created_at=datetime.now(),
            metadata={}
        )
    
    @property
    def memory_type(self) -> str:
        return self._base.memory_type
    
    @property
    def created_at(self) -> datetime:
        return self._base.created_at
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._base.metadata

@dataclass
class ContextMemory:
    """Memory for maintaining context across operations."""
    content: Any
    context_type: str  # e.g., "project", "file", "function"
    scope: str
    _base: Memory = field(init=False)
    
    def __post_init__(self):
        self._base = Memory(
            content=self.content,
            memory_type="context",
            created_at=datetime.now(),
            metadata={}
        )
    
    @property
    def memory_type(self) -> str:
        return self._base.memory_type
    
    @property
    def created_at(self) -> datetime:
        return self._base.created_at
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._base.metadata

class MemoryManager:
    """Manages different types of memory for the agent."""
    
    def __init__(self):
        """Initialize the memory manager."""
        self.memories: Dict[str, List[Memory | CodeMemory | ContextMemory]] = {}
    
    def add_memory(self, memory: Memory | CodeMemory | ContextMemory) -> None:
        """Add a memory item."""
        if memory.memory_type not in self.memories:
            self.memories[memory.memory_type] = []
        self.memories[memory.memory_type].append(memory)
    
    def get_memories_by_type(self, memory_type: str) -> List[Memory | CodeMemory | ContextMemory]:
        """Get all memories of a specific type."""
        return self.memories.get(memory_type, [])
    
    def get_recent_memories(self, limit: int = 10, memory_type: Optional[str] = None) -> List[Memory | CodeMemory | ContextMemory]:
        """Get the most recent memories, optionally filtered by type."""
        all_memories = []
        for mtype, memories in self.memories.items():
            if memory_type is None or mtype == memory_type:
                all_memories.extend(memories)
        
        # Sort by created_at in reverse order (most recent first)
        all_memories.sort(key=lambda x: x.created_at, reverse=True)
        return all_memories[:limit]
    
    def add_code_memory(
        self,
        file_path: str,
        code_type: str,
        content: Any,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """Add a code memory."""
        memory = CodeMemory(
            content=content,
            file_path=file_path,
            code_type=code_type,
            dependencies=dependencies or []
        )
        self.add_memory(memory)
    
    def add_context_memory(
        self,
        context_type: str,
        scope: str,
        content: Any
    ) -> None:
        """Add a context memory."""
        memory = ContextMemory(
            content=content,
            context_type=context_type,
            scope=scope
        )
        self.add_memory(memory)
    
    def update_working_memory(self, key: str, value: Any) -> None:
        """Update working memory with a key-value pair."""
        memory = Memory(
            content={key: value},
            memory_type="working"
        )
        self.add_memory(memory)
    
    def get_working_memory(self, key: Optional[str] = None) -> Union[Dict[str, Any], Any, None]:
        """Get the current working memory state or a specific value.
        
        Args:
            key: Optional key to retrieve specific value. If None, returns entire state.
            
        Returns:
            If key is None, returns entire working memory state dict.
            If key is provided, returns the value for that key or None if not found.
        """
        working_memories = self.get_memories_by_type("working")
        state = {}
        for memory in working_memories:
            if isinstance(memory.content, dict):
                state.update(memory.content)
        
        if key is not None:
            return state.get(key)
        return state
    
    def clear_working_memory(self) -> None:
        """Clear all working memory."""
        self.memories["working"] = []
    
    def get_related_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Memory | CodeMemory | ContextMemory]:
        """Get memories related to a query using basic text matching."""
        all_memories = []
        query_terms = set(query.lower().split())
        
        for mtype, memories in self.memories.items():
            if memory_type is None or mtype == memory_type:
                for memory in memories:
                    # Convert content to string and check if all query terms are present
                    content_str = str(memory.content).lower()
                    if all(term in content_str for term in query_terms):
                        all_memories.append(memory)
        
        # Sort by relevance (number of query terms matched) and recency
        all_memories.sort(
            key=lambda x: (
                sum(term in str(x.content).lower() for term in query_terms),
                x.created_at
            ),
            reverse=True
        )
        return all_memories[:limit]
    
    def save_to_file(self, file_path: str) -> None:
        """Save memories to a file."""
        serialized_memories = {
            mtype: [
                {
                    "content": memory.content if isinstance(memory.content, (str, int, float, bool)) else json.dumps(memory.content),
                    "created_at": memory.created_at.isoformat(),
                    "memory_type": memory.memory_type,
                    "metadata": memory.metadata,
                    **({"file_path": memory.file_path, "code_type": memory.code_type, "dependencies": memory.dependencies} if isinstance(memory, CodeMemory) else {}),
                    **({"context_type": memory.context_type, "scope": memory.scope} if isinstance(memory, ContextMemory) else {})
                }
                for memory in mlist
            ]
            for mtype, mlist in self.memories.items()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(serialized_memories, f)
    
    def load_from_file(self, file_path: str) -> None:
        """Load memories from a file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        
        self.memories.clear()
        for mtype, memories in data.items():
            for mem_data in memories:
                # Try to parse content as JSON if it looks like a dict/list
                content = mem_data["content"]
                if isinstance(content, str):
                    if content.startswith(("{", "[")):
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not valid JSON
                
                created_at = datetime.fromisoformat(mem_data["created_at"])
                metadata = mem_data.get("metadata", {})
                
                if mtype == "code":
                    memory = CodeMemory(
                        content=content,
                        file_path=mem_data["file_path"],
                        code_type=mem_data["code_type"],
                        dependencies=mem_data.get("dependencies", [])
                    )
                elif mtype == "context":
                    memory = ContextMemory(
                        content=content,
                        context_type=mem_data["context_type"],
                        scope=mem_data["scope"]
                    )
                else:
                    memory = Memory(
                        content=content,
                        memory_type=mtype,
                        created_at=created_at,
                        metadata=metadata
                    )
                
                self.add_memory(memory) 
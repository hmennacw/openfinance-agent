from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class Memory:
    """Base class for memory items."""
    content: Any
    created_at: datetime = field(default_factory=datetime.now)
    memory_type: str = field(default="generic")
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeMemory(Memory):
    """Memory specific to code generation."""
    file_path: str
    code_type: str  # e.g., "handler", "usecase", "model"
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.memory_type = "code"

@dataclass
class ContextMemory(Memory):
    """Memory for maintaining context across operations."""
    context_type: str  # e.g., "project", "file", "function"
    scope: str
    
    def __post_init__(self):
        self.memory_type = "context"

class MemoryManager:
    """Manages different types of memory for the agent."""
    
    def __init__(self):
        self.memories: Dict[str, List[Memory]] = {
            "code": [],
            "context": [],
            "generic": []
        }
        self.working_memory: Dict[str, Any] = {}
    
    def add_memory(self, memory: Memory) -> None:
        """Add a new memory item."""
        self.memories[memory.memory_type].append(memory)
    
    def get_memories_by_type(self, memory_type: str) -> List[Memory]:
        """Retrieve all memories of a specific type."""
        return self.memories.get(memory_type, [])
    
    def get_recent_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Get the most recent memories, optionally filtered by type."""
        memories = []
        if memory_type:
            memories = self.memories.get(memory_type, [])
        else:
            for mem_list in self.memories.values():
                memories.extend(mem_list)
        
        return sorted(
            memories,
            key=lambda x: x.created_at,
            reverse=True
        )[:limit]
    
    def add_code_memory(
        self,
        file_path: str,
        code_type: str,
        content: str,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """Add a new code memory."""
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
        """Add a new context memory."""
        memory = ContextMemory(
            content=content,
            context_type=context_type,
            scope=scope
        )
        self.add_memory(memory)
    
    def update_working_memory(self, key: str, value: Any) -> None:
        """Update the working memory with new information."""
        self.working_memory[key] = value
    
    def get_working_memory(self, key: str) -> Optional[Any]:
        """Retrieve a value from working memory."""
        return self.working_memory.get(key)
    
    def clear_working_memory(self) -> None:
        """Clear the working memory."""
        self.working_memory.clear()
    
    def get_related_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Memory]:
        """
        Get memories related to a query.
        This is a simple implementation that could be enhanced with
        embedding-based similarity search.
        """
        memories = self.get_memories_by_type(memory_type) if memory_type else [
            m for mlist in self.memories.values() for m in mlist
        ]
        
        # Simple keyword matching for now
        # Could be enhanced with embeddings and similarity search
        query_terms = query.lower().split()
        scored_memories = []
        
        for memory in memories:
            score = 0
            memory_content = str(memory.content).lower()
            
            for term in query_terms:
                if term in memory_content:
                    score += 1
            
            if score > 0:
                scored_memories.append((score, memory))
        
        return [
            m for _, m in sorted(
                scored_memories,
                key=lambda x: (x[0], x[1].created_at),
                reverse=True
            )
        ][:limit]
    
    def save_to_file(self, file_path: str) -> None:
        """Save memories to a file."""
        serialized_memories = {
            mtype: [
                {
                    "content": str(m.content),
                    "created_at": m.created_at.isoformat(),
                    "memory_type": m.memory_type,
                    "metadata": m.metadata,
                    **{
                        k: v for k, v in m.__dict__.items()
                        if k not in ["content", "created_at", "memory_type", "metadata"]
                    }
                }
                for m in mlist
            ]
            for mtype, mlist in self.memories.items()
        }
        
        with open(file_path, "w") as f:
            json.dump(serialized_memories, f, indent=2)
    
    def load_from_file(self, file_path: str) -> None:
        """Load memories from a file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        
        self.memories.clear()
        for mtype, mlist in data.items():
            self.memories[mtype] = []
            for mdata in mlist:
                created_at = datetime.fromisoformat(mdata.pop("created_at"))
                memory_type = mdata.pop("memory_type")
                content = mdata.pop("content")
                metadata = mdata.pop("metadata")
                
                if memory_type == "code":
                    memory = CodeMemory(
                        content=content,
                        created_at=created_at,
                        metadata=metadata,
                        **mdata
                    )
                elif memory_type == "context":
                    memory = ContextMemory(
                        content=content,
                        created_at=created_at,
                        metadata=metadata,
                        **mdata
                    )
                else:
                    memory = Memory(
                        content=content,
                        created_at=created_at,
                        memory_type=memory_type,
                        metadata=metadata
                    )
                
                self.memories[memory_type].append(memory) 
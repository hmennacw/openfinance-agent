import pytest
from datetime import datetime
from pathlib import Path
import json

from src.cognitive.memory import (
    Memory,
    CodeMemory,
    ContextMemory,
    MemoryManager
)

def test_memory_creation():
    """Test basic memory creation."""
    memory = Memory(content="test content")
    assert memory.content == "test content"
    assert memory.memory_type == "generic"
    assert isinstance(memory.created_at, datetime)

def test_code_memory_creation():
    """Test code memory creation."""
    memory = CodeMemory(
        content="package main",
        file_path="main.go",
        code_type="main"
    )
    assert memory.content == "package main"
    assert memory.file_path == "main.go"
    assert memory.code_type == "main"
    assert memory.memory_type == "code"

def test_context_memory_creation():
    """Test context memory creation."""
    memory = ContextMemory(
        content={"key": "value"},
        context_type="test",
        scope="global"
    )
    assert memory.content == {"key": "value"}
    assert memory.context_type == "test"
    assert memory.scope == "global"
    assert memory.memory_type == "context"

class TestMemoryManager:
    """Test suite for MemoryManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a memory manager instance."""
        return MemoryManager()
    
    def test_add_memory(self, manager):
        """Test adding a memory item."""
        memory = Memory(content="test")
        manager.add_memory(memory)
        assert memory in manager.memories["generic"]
    
    def test_get_memories_by_type(self, manager):
        """Test retrieving memories by type."""
        memory = Memory(content="test")
        manager.add_memory(memory)
        memories = manager.get_memories_by_type("generic")
        assert memory in memories
    
    def test_get_recent_memories(self, manager):
        """Test retrieving recent memories."""
        for i in range(1, 16):  # Create memories with IDs 1-15
            manager.add_memory(Memory(content=f"test_{i}"))
        
        recent = manager.get_recent_memories(limit=10)
        assert len(recent) == 10
        # Most recent should be last added
        assert recent[0].content == "test_15"
    
    def test_add_code_memory(self, manager, sample_code_memory):
        """Test adding a code memory."""
        manager.add_code_memory(
            file_path=sample_code_memory["file_path"],
            code_type=sample_code_memory["code_type"],
            content=sample_code_memory["content"],
            dependencies=sample_code_memory["dependencies"]
        )
        
        memories = manager.get_memories_by_type("code")
        assert len(memories) == 1
        memory = memories[0]
        assert memory.file_path == sample_code_memory["file_path"]
        assert memory.code_type == sample_code_memory["code_type"]
        assert memory.content == sample_code_memory["content"]
        assert memory.dependencies == sample_code_memory["dependencies"]
    
    def test_add_context_memory(self, manager):
        """Test adding a context memory."""
        context = {"endpoint": "/users", "method": "GET"}
        manager.add_context_memory(
            context_type="api",
            scope="endpoint",
            content=context
        )
        
        memories = manager.get_memories_by_type("context")
        assert len(memories) == 1
        memory = memories[0]
        assert memory.context_type == "api"
        assert memory.scope == "endpoint"
        assert memory.content == context
    
    def test_working_memory(self, manager):
        """Test working memory operations."""
        # Test setting and getting individual values
        manager.update_working_memory("key1", "value1")
        manager.update_working_memory("key2", "value2")
        
        # Test getting specific keys
        assert manager.get_working_memory("key1") == "value1"
        assert manager.get_working_memory("key2") == "value2"
        assert manager.get_working_memory("nonexistent") is None
        
        # Test getting entire state
        state = manager.get_working_memory()
        assert state == {"key1": "value1", "key2": "value2"}
        
        # Test clearing memory
        manager.clear_working_memory()
        assert manager.get_working_memory() == {}
        assert manager.get_working_memory("key1") is None
    
    def test_get_related_memories(self, manager):
        """Test retrieving related memories."""
        manager.add_memory(Memory(content="test user api"))
        manager.add_memory(Memory(content="test product api"))
        manager.add_memory(Memory(content="something else"))
        
        related = manager.get_related_memories("test api")
        assert len(related) == 2
        assert all("test" in m.content and "api" in m.content for m in related)
    
    def test_save_and_load(self, manager, temp_storage_path):
        """Test saving and loading memories."""
        # Add some test memories
        manager.add_code_memory(
            file_path="test.go",
            code_type="test",
            content="package test"
        )
        manager.add_context_memory(
            context_type="test",
            scope="global",
            content={"test": True}
        )
        
        # Save memories
        storage_file = temp_storage_path / "memories.json"
        manager.save_to_file(str(storage_file))
        
        # Create new manager and load memories
        new_manager = MemoryManager()
        new_manager.load_from_file(str(storage_file))
        
        # Verify memories were loaded correctly
        assert len(new_manager.memories["code"]) == 1
        assert len(new_manager.memories["context"]) == 1
        
        code_memory = new_manager.memories["code"][0]
        assert code_memory.file_path == "test.go"
        assert code_memory.code_type == "test"
        assert code_memory.content == "package test"
        
        context_memory = new_manager.memories["context"][0]
        assert context_memory.context_type == "test"
        assert context_memory.scope == "global"
        assert context_memory.content == {"test": True} 
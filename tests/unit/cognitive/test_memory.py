import pytest
from datetime import datetime
from pathlib import Path
import json
import os
import tempfile
import shutil

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
    assert isinstance(memory.metadata, dict)
    assert len(memory.metadata) == 0

def test_memory_with_metadata():
    """Test memory creation with metadata."""
    metadata = {"key": "value"}
    memory = Memory(
        content="test content",
        memory_type="custom",
        metadata=metadata
    )
    assert memory.memory_type == "custom"
    assert memory.metadata == metadata

def test_code_memory_creation():
    """Test code memory creation."""
    memory = CodeMemory(
        content="package main",
        file_path="main.go",
        code_type="main",
        dependencies=["fmt"]
    )
    assert memory.content == "package main"
    assert memory.file_path == "main.go"
    assert memory.code_type == "main"
    assert memory.dependencies == ["fmt"]
    assert memory.memory_type == "code"
    assert isinstance(memory.created_at, datetime)
    assert isinstance(memory.metadata, dict)

def test_code_memory_default_dependencies():
    """Test code memory creation with default dependencies."""
    memory = CodeMemory(
        content="package main",
        file_path="main.go",
        code_type="main"
    )
    assert memory.dependencies == []

def test_code_memory_properties():
    """Test code memory property access."""
    memory = CodeMemory(
        content="test",
        file_path="test.go",
        code_type="test"
    )
    assert memory.memory_type == "code"
    assert isinstance(memory.created_at, datetime)
    assert isinstance(memory.metadata, dict)

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
    assert isinstance(memory.created_at, datetime)
    assert isinstance(memory.metadata, dict)

def test_context_memory_properties():
    """Test context memory property access."""
    memory = ContextMemory(
        content="test",
        context_type="test",
        scope="local"
    )
    assert memory.memory_type == "context"
    assert isinstance(memory.created_at, datetime)
    assert isinstance(memory.metadata, dict)

class TestMemoryManager:
    """Test suite for MemoryManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a memory manager instance."""
        return MemoryManager()
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary directory for storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_add_memory(self, manager):
        """Test adding a memory item."""
        memory = Memory(content="test")
        manager.add_memory(memory)
        assert memory in manager.memories["generic"]
    
    def test_add_code_memory(self, manager):
        """Test adding a code memory."""
        manager.add_code_memory(
            file_path="test.go",
            code_type="test",
            content="package main",
            dependencies=["fmt"]
        )
        
        memories = manager.get_memories_by_type("code")
        assert len(memories) == 1
        memory = memories[0]
        assert isinstance(memory, CodeMemory)
        assert memory.file_path == "test.go"
        assert memory.code_type == "test"
        assert memory.content == "package main"
        assert memory.dependencies == ["fmt"]
    
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
        assert isinstance(memory, ContextMemory)
        assert memory.context_type == "api"
        assert memory.scope == "endpoint"
        assert memory.content == context
    
    def test_get_memories_by_type(self, manager):
        """Test retrieving memories by type."""
        memory1 = Memory(content="test1", memory_type="type1")
        memory2 = Memory(content="test2", memory_type="type2")
        manager.add_memory(memory1)
        manager.add_memory(memory2)
        
        type1_memories = manager.get_memories_by_type("type1")
        assert len(type1_memories) == 1
        assert type1_memories[0] == memory1
        
        # Test non-existent type
        assert manager.get_memories_by_type("nonexistent") == []
    
    def test_get_recent_memories(self, manager):
        """Test retrieving recent memories."""
        # Add memories of different types
        manager.add_memory(Memory(content="test1", memory_type="type1"))
        manager.add_memory(Memory(content="test2", memory_type="type2"))
        manager.add_memory(Memory(content="test3", memory_type="type1"))
        
        # Test getting all recent memories
        recent = manager.get_recent_memories(limit=2)
        assert len(recent) == 2
        assert recent[0].content == "test3"  # Most recent first
        assert recent[1].content == "test2"
        
        # Test getting recent memories by type
        type1_recent = manager.get_recent_memories(limit=2, memory_type="type1")
        assert len(type1_recent) == 2
        assert all(m.memory_type == "type1" for m in type1_recent)
    
    def test_update_working_memory(self, manager):
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
        # Add test memories
        manager.add_memory(Memory(content="test user api"))
        manager.add_memory(Memory(content="test product api"))
        manager.add_memory(Memory(content="something else"))
        
        # Test searching with multiple terms
        related = manager.get_related_memories("test api")
        assert len(related) == 2
        assert all("test" in m.content and "api" in m.content for m in related)
        
        # Test searching with memory type filter
        manager.add_memory(Memory(content="test api", memory_type="filtered"))
        filtered = manager.get_related_memories("test api", memory_type="filtered")
        assert len(filtered) == 1
        assert filtered[0].memory_type == "filtered"
        
        # Test with limit
        limited = manager.get_related_memories("test", limit=2)
        assert len(limited) == 2
    
    def test_save_and_load(self, manager, temp_storage_path):
        """Test saving and loading memories."""
        # Add test memories
        manager.add_code_memory(
            file_path="test.go",
            code_type="test",
            content="package test",
            dependencies=["fmt"]
        )
        manager.add_context_memory(
            context_type="test",
            scope="global",
            content={"test": True}
        )
        manager.add_memory(Memory(
            content="test",
            memory_type="custom",
            metadata={"key": "value"}
        ))
        
        # Save memories
        storage_file = os.path.join(temp_storage_path, "memories.json")
        manager.save_to_file(storage_file)
        
        # Create new manager and load memories
        new_manager = MemoryManager()
        new_manager.load_from_file(storage_file)
        
        # Verify memories were loaded correctly
        assert len(new_manager.memories["code"]) == 1
        assert len(new_manager.memories["context"]) == 1
        assert len(new_manager.memories["custom"]) == 1
        
        # Verify code memory
        code_memory = new_manager.memories["code"][0]
        assert isinstance(code_memory, CodeMemory)
        assert code_memory.file_path == "test.go"
        assert code_memory.code_type == "test"
        assert code_memory.content == "package test"
        assert code_memory.dependencies == ["fmt"]
        
        # Verify context memory
        context_memory = new_manager.memories["context"][0]
        assert isinstance(context_memory, ContextMemory)
        assert context_memory.context_type == "test"
        assert context_memory.scope == "global"
        assert context_memory.content == {"test": True}
        
        # Verify custom memory
        custom_memory = new_manager.memories["custom"][0]
        assert isinstance(custom_memory, Memory)
        assert custom_memory.content == "test"
        assert custom_memory.metadata == {"key": "value"}
    
    def test_save_with_complex_content(self, manager, temp_storage_path):
        """Test saving memories with complex content types."""
        complex_content = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "bool": True,
            "int": 42,
            "float": 3.14
        }
        
        manager.add_memory(Memory(content=complex_content))
        
        storage_file = os.path.join(temp_storage_path, "memories.json")
        manager.save_to_file(storage_file)
        
        # Load and verify
        new_manager = MemoryManager()
        new_manager.load_from_file(storage_file)
        
        loaded_memory = new_manager.memories["generic"][0]
        assert loaded_memory.content == complex_content
    
    def test_load_with_invalid_json(self, manager, temp_storage_path):
        """Test loading from invalid JSON file."""
        storage_file = os.path.join(temp_storage_path, "invalid.json")
        
        # Create invalid JSON file
        with open(storage_file, "w") as f:
            f.write("invalid json content")
        
        # Should not raise exception
        manager.load_from_file(storage_file)
        assert len(manager.memories) == 0
    
    def test_save_with_directory_creation(self, manager, temp_storage_path):
        """Test saving to a new directory."""
        nested_dir = os.path.join(temp_storage_path, "nested", "dir")
        storage_file = os.path.join(nested_dir, "memories.json")
        
        manager.add_memory(Memory(content="test"))
        manager.save_to_file(storage_file)
        
        assert os.path.exists(storage_file)
    
    def test_load_nonexistent_file(self, manager):
        """Test loading from a nonexistent file."""
        manager.load_from_file("nonexistent.json")
        assert len(manager.memories) == 0
    
    def test_memory_type_initialization(self, manager):
        """Test that memory types are initialized on first use."""
        memory_type = "new_type"
        assert memory_type not in manager.memories
        
        manager.add_memory(Memory(content="test", memory_type=memory_type))
        assert memory_type in manager.memories
        assert len(manager.memories[memory_type]) == 1
    
    def test_load_with_invalid_content_json(self, manager, temp_storage_path):
        """Test loading file with invalid JSON in content field."""
        # Create a valid JSON file with invalid JSON string in content
        storage_file = os.path.join(temp_storage_path, "test.json")
        test_data = {
            "generic": [{
                "content": "{invalid_json",  # Invalid JSON string
                "memory_type": "generic",
                "created_at": datetime.now().isoformat(),
                "metadata": {}
            }]
        }
        
        with open(storage_file, "w") as f:
            json.dump(test_data, f)
        
        # Should load successfully, keeping the invalid JSON as a string
        manager.load_from_file(storage_file)
        assert len(manager.memories["generic"]) == 1
        assert manager.memories["generic"][0].content == "{invalid_json" 
import pytest
from datetime import datetime
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock
import logging

from src.cognitive.planner import (
    TaskStatus,
    Task,
    TaskExecutionContext,
    TaskPlanner
)

def test_task_status_enum():
    """Test TaskStatus enum values."""
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.IN_PROGRESS.value == "in_progress"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value == "failed"
    assert TaskStatus.BLOCKED.value == "blocked"

def test_task_creation():
    """Test basic task creation."""
    task = Task(
        id="test_task",
        name="Test Task",
        description="A test task",
        dependencies=[]
    )
    assert task.id == "test_task"
    assert task.name == "Test Task"
    assert task.description == "A test task"
    assert task.dependencies == []
    assert task.status == TaskStatus.PENDING
    assert isinstance(task.created_at, datetime)
    assert task.started_at is None
    assert task.completed_at is None
    assert task.error is None
    assert isinstance(task.metadata, dict)
    assert task.subtasks == []
    assert task.parent_id is None
    assert task.priority == 0

def test_task_equality():
    """Test task equality based on ID."""
    task1 = Task(id="test", name="Test", description="Test", dependencies=[])
    task2 = Task(id="test", name="Different", description="Different", dependencies=[])
    task3 = Task(id="different", name="Test", description="Test", dependencies=[])
    
    assert task1 == task2  # Same ID
    assert task1 != task3  # Different ID
    assert task1 != "test"  # Different type

def test_task_hash():
    """Test task hash based on ID."""
    task = Task(id="test", name="Test", description="Test", dependencies=[])
    assert hash(task) == hash("test")

def test_task_add_subtask():
    """Test adding subtasks to a task."""
    parent = Task(
        id="parent",
        name="Parent Task",
        description="Parent task",
        dependencies=[]
    )
    
    child = Task(
        id="child",
        name="Child Task",
        description="Child task",
        dependencies=[]
    )
    
    # Test adding subtask
    updated_parent = parent.add_subtask(child)
    
    # Verify parent is unchanged (immutable)
    assert len(parent.subtasks) == 0
    
    # Verify new parent has subtask
    assert len(updated_parent.subtasks) == 1
    assert updated_parent.subtasks[0].id == "child"
    assert updated_parent.subtasks[0].parent_id == "parent"
    
    # Verify all other fields are preserved
    assert updated_parent.id == parent.id
    assert updated_parent.name == parent.name
    assert updated_parent.description == parent.description
    assert updated_parent.status == parent.status

def test_task_update_status():
    """Test updating task status."""
    task = Task(
        id="test",
        name="Test Task",
        description="Test task",
        dependencies=[]
    )
    
    # Test starting task
    in_progress = task.update_status(TaskStatus.IN_PROGRESS)
    assert in_progress.status == TaskStatus.IN_PROGRESS
    assert in_progress.started_at is not None
    assert in_progress.completed_at is None
    
    # Test completing task
    completed = in_progress.update_status(TaskStatus.COMPLETED)
    assert completed.status == TaskStatus.COMPLETED
    assert completed.started_at == in_progress.started_at
    assert completed.completed_at is not None
    
    # Test failing task with error
    failed = task.update_status(TaskStatus.FAILED, "Test error")
    assert failed.status == TaskStatus.FAILED
    assert failed.error == "Test error"
    assert failed.completed_at is not None

def test_task_execution_context():
    """Test TaskExecutionContext functionality."""
    task = Task(id="test", name="Test", description="Test", dependencies=[])
    memory_manager = MagicMock()
    
    context = TaskExecutionContext(task=task, memory_manager=memory_manager)
    
    # Test variable management
    context.set_variable("key", "value")
    assert context.get_variable("key") == "value"
    assert context.get_variable("nonexistent") is None
    assert context.get_variable("nonexistent", "default") == "default"

class TestTaskPlanner:
    """Test suite for TaskPlanner."""
    
    @pytest.fixture
    def planner(self):
        """Create a task planner instance."""
        return TaskPlanner()
    
    @pytest.fixture
    def memory_manager(self):
        """Create a mock memory manager."""
        return MagicMock()
    
    def test_initialization(self, planner):
        """Test planner initialization."""
        assert isinstance(planner.tasks, dict)
        assert isinstance(planner.handlers, dict)
        assert isinstance(planner.logger, logging.Logger)
    
    def test_register_task_handler(self, planner):
        """Test registering a task handler."""
        async def handler(context: TaskExecutionContext):
            pass
        
        planner.register_task_handler("test_type", handler)
        assert "test_type" in planner.handlers
        assert planner.handlers["test_type"] == handler
    
    def test_create_task(self, planner):
        """Test task creation."""
        task = planner.create_task(
            id="test",
            name="Test Task",
            description="A test task",
            dependencies=["dep1"],
            metadata={"key": "value"},
            priority=1
        )
        
        assert task.id == "test"
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.dependencies == ["dep1"]
        assert task.metadata == {"key": "value"}
        assert task.priority == 1
        assert task in planner.tasks.values()
    
    def test_get_task(self, planner):
        """Test retrieving a task."""
        task = planner.create_task(
            id="test",
            name="Test Task",
            description="A test task"
        )
        
        assert planner.get_task("test") == task
        assert planner.get_task("nonexistent") is None
    
    def test_get_all_tasks(self, planner):
        """Test retrieving all tasks."""
        task1 = planner.create_task("task1", "Task 1", "First task")
        task2 = planner.create_task("task2", "Task 2", "Second task")
        
        all_tasks = planner.get_all_tasks()
        assert len(all_tasks) == 2
        assert task1 in all_tasks
        assert task2 in all_tasks
    
    def test_get_pending_tasks(self, planner):
        """Test retrieving pending tasks."""
        pending = planner.create_task("pending", "Pending", "Pending task")
        completed = planner.create_task("completed", "Completed", "Completed task")
        
        # Update completed task status
        planner.tasks["completed"] = completed.update_status(TaskStatus.COMPLETED)
        
        pending_tasks = planner.get_pending_tasks()
        assert len(pending_tasks) == 1
        assert pending_tasks[0] == pending
    
    def test_get_blocked_tasks(self, planner):
        """Test retrieving blocked tasks."""
        # Create tasks with dependencies
        dep_task = planner.create_task("dep", "Dependency", "Dependency task")
        blocked_task = planner.create_task(
            "blocked",
            "Blocked",
            "Blocked task",
            dependencies=["dep"]
        )
        
        # Initially dep_task is pending, so blocked_task is blocked
        blocked_tasks = planner.get_blocked_tasks()
        assert len(blocked_tasks) == 1
        assert blocked_tasks[0] == blocked_task
        
        # Complete dep_task
        planner.tasks["dep"] = dep_task.update_status(TaskStatus.COMPLETED)
        
        # Now blocked_task should not be blocked
        assert len(planner.get_blocked_tasks()) == 0
    
    def test_can_execute_task(self, planner):
        """Test task execution conditions."""
        # Create tasks with dependencies
        dep_task = planner.create_task("dep", "Dependency", "Dependency task")
        dependent_task = planner.create_task(
            "dependent",
            "Dependent",
            "Dependent task",
            dependencies=["dep"]
        )
        
        # Initially dependent_task cannot be executed
        assert not planner._can_execute_task(dependent_task)
        
        # Complete dependency
        planner.tasks["dep"] = dep_task.update_status(TaskStatus.COMPLETED)
        
        # Now dependent_task can be executed
        assert planner._can_execute_task(dependent_task)
        
        # But if dependent_task is not pending, it cannot be executed
        non_pending = dependent_task.update_status(TaskStatus.IN_PROGRESS)
        assert not planner._can_execute_task(non_pending)
    
    def test_get_next_task(self, planner):
        """Test next task selection."""
        # Create tasks with different priorities
        low_priority = planner.create_task(
            "low",
            "Low Priority",
            "Low priority task",
            priority=1
        )
        high_priority = planner.create_task(
            "high",
            "High Priority",
            "High priority task",
            priority=2
        )
        
        # Should select high priority task
        next_task = planner._get_next_task()
        assert next_task == high_priority
        
        # Complete high priority task
        planner.tasks["high"] = high_priority.update_status(TaskStatus.COMPLETED)
        
        # Should select low priority task
        next_task = planner._get_next_task()
        assert next_task == low_priority
        
        # Complete all tasks
        planner.tasks["low"] = low_priority.update_status(TaskStatus.COMPLETED)
        
        # Should return None when no executable tasks
        assert planner._get_next_task() is None
    
    @pytest.mark.asyncio
    async def test_execute_task(self, planner):
        """Test task execution."""
        executed = False
        
        async def test_handler(context: TaskExecutionContext):
            nonlocal executed
            executed = True
        
        task = planner.create_task(
            id="test",
            name="Test Task",
            description="Test task",
            metadata={"type": "test_type"}
        )
        
        planner.register_task_handler("test_type", test_handler)
        
        context = TaskExecutionContext(
            task=task,
            memory_manager=MagicMock()
        )
        
        await planner.execute_task(task, context)
        assert executed
        
        # Verify task status updates
        updated_task = planner.get_task("test")
        assert updated_task.status == TaskStatus.COMPLETED
        assert updated_task.started_at is not None
        assert updated_task.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, planner):
        """Test task execution failure."""
        async def failing_handler(context: TaskExecutionContext):
            raise ValueError("Test error")
        
        task = planner.create_task(
            id="test",
            name="Test Task",
            description="Test task",
            metadata={"type": "test_type"}
        )
        
        planner.register_task_handler("test_type", failing_handler)
        
        context = TaskExecutionContext(
            task=task,
            memory_manager=MagicMock()
        )
        
        with pytest.raises(ValueError):
            await planner.execute_task(task, context)
        
        # Verify task status updates
        failed_task = planner.get_task("test")
        assert failed_task.status == TaskStatus.FAILED
        assert failed_task.error == "Test error"
        assert failed_task.started_at is not None
        assert failed_task.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_execute_all_tasks(self, planner, memory_manager):
        """Test execution of all tasks."""
        executed_tasks = []
        
        async def test_handler(context: TaskExecutionContext):
            executed_tasks.append(context.task.id)
        
        # Create tasks with dependencies
        task1 = planner.create_task(
            id="task1",
            name="Task 1",
            description="First task",
            metadata={"type": "test_type"}
        )
        
        task2 = planner.create_task(
            id="task2",
            name="Task 2",
            description="Second task",
            dependencies=["task1"],
            metadata={"type": "test_type"}
        )
        
        planner.register_task_handler("test_type", test_handler)
        
        await planner.execute_all_tasks(memory_manager)
        
        assert executed_tasks == ["task1", "task2"]
        assert all(
            task.status == TaskStatus.COMPLETED
            for task in planner.get_all_tasks()
        )
    
    @pytest.mark.asyncio
    async def test_execute_all_tasks_with_failure(self, planner, memory_manager):
        """Test execution with task failure."""
        async def failing_handler(context: TaskExecutionContext):
            raise ValueError("Test error")
        
        task1 = planner.create_task(
            id="task1",
            name="Task 1",
            description="First task",
            metadata={"type": "failing_type"}
        )
        
        task2 = planner.create_task(
            id="task2",
            name="Task 2",
            description="Second task",
            dependencies=["task1"],
            metadata={"type": "failing_type"}
        )
        
        planner.register_task_handler("failing_type", failing_handler)
        
        with pytest.raises(ValueError):
            await planner.execute_all_tasks(memory_manager)
        
        # Verify task statuses
        failed_task = planner.get_task("task1")
        blocked_task = planner.get_task("task2")
        
        assert failed_task.status == TaskStatus.FAILED
        assert blocked_task.status == TaskStatus.BLOCKED
    
    def test_reset(self, planner):
        """Test resetting planner state."""
        planner.create_task("test", "Test", "Test task")
        assert len(planner.tasks) == 1
        
        planner.reset()
        assert len(planner.tasks) == 0
    
    def test_get_execution_plan(self, planner):
        """Test execution plan generation."""
        # Create tasks with dependencies
        task1 = planner.create_task(
            id="task1",
            name="Task 1",
            description="First task"
        )
        
        task2 = planner.create_task(
            id="task2",
            name="Task 2",
            description="Second task",
            dependencies=["task1"]
        )
        
        task3 = planner.create_task(
            id="task3",
            name="Task 3",
            description="Third task",
            dependencies=["task1"]
        )
        
        task4 = planner.create_task(
            id="task4",
            name="Task 4",
            description="Fourth task",
            dependencies=["task2", "task3"]
        )
        
        plan = planner.get_execution_plan()
        
        # Verify plan levels
        assert len(plan) == 3  # Should have 3 levels
        assert task1 in plan[0]  # Level 0: task1
        assert all(t in plan[1] for t in [task2, task3])  # Level 1: task2, task3
        assert task4 in plan[2]  # Level 2: task4
    
    @pytest.mark.asyncio
    async def test_execute_all_tasks_with_blocked(self, planner, memory_manager):
        """Test execution with blocked tasks."""
        async def test_handler(context: TaskExecutionContext):
            pass
        
        # Create tasks with missing dependency
        task = planner.create_task(
            id="task",
            name="Task",
            description="Task with missing dependency",
            dependencies=["missing"],
            metadata={"type": "test_type"}
        )
        
        planner.register_task_handler("test_type", test_handler)
        
        # Execute should handle blocked tasks
        await planner.execute_all_tasks(memory_manager)
        
        blocked_task = planner.get_task("task")
        assert blocked_task.status == TaskStatus.BLOCKED 
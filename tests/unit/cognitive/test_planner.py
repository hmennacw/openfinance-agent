import pytest
from datetime import datetime
import asyncio
from typing import Dict, Any

from src.cognitive.planner import (
    TaskStatus,
    Task,
    TaskExecutionContext,
    TaskPlanner
)

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
    assert task.status == TaskStatus.PENDING
    assert isinstance(task.created_at, datetime)

def test_task_status_update():
    """Test task status updates."""
    task = Task(
        id="test_task",
        name="Test Task",
        description="A test task",
        dependencies=[]
    )
    
    # Test starting the task
    task.update_status(TaskStatus.IN_PROGRESS)
    assert task.status == TaskStatus.IN_PROGRESS
    assert task.started_at is not None
    assert task.completed_at is None
    
    # Test completing the task
    task.update_status(TaskStatus.COMPLETED)
    assert task.status == TaskStatus.COMPLETED
    assert task.completed_at is not None
    
    # Test failing the task
    error_msg = "Test error"
    task.update_status(TaskStatus.FAILED, error_msg)
    assert task.status == TaskStatus.FAILED
    assert task.error == error_msg

def test_task_subtasks():
    """Test task subtask management."""
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
    
    parent.add_subtask(child)
    assert child in parent.subtasks
    assert child.parent_id == parent.id

class TestTaskPlanner:
    """Test suite for TaskPlanner."""
    
    @pytest.fixture
    def planner(self):
        """Create a task planner instance."""
        return TaskPlanner()
    
    def test_create_task(self, planner):
        """Test task creation in planner."""
        task = planner.create_task(
            task_id="test",
            name="Test Task",
            description="A test task"
        )
        assert task.id == "test"
        assert planner.get_task("test") == task
    
    def test_get_pending_tasks(self, planner):
        """Test retrieving pending tasks."""
        task1 = planner.create_task(
            task_id="task1",
            name="Task 1",
            description="Task 1"
        )
        task2 = planner.create_task(
            task_id="task2",
            name="Task 2",
            description="Task 2"
        )
        task2.update_status(TaskStatus.IN_PROGRESS)
        
        pending = planner.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0] == task1
    
    def test_get_blocked_tasks(self, planner):
        """Test retrieving blocked tasks."""
        task = planner.create_task(
            task_id="task",
            name="Task",
            description="Task"
        )
        task.update_status(TaskStatus.BLOCKED)
        
        blocked = planner.get_blocked_tasks()
        assert len(blocked) == 1
        assert blocked[0] == task
    
    def test_task_dependencies(self, planner):
        """Test task dependency handling."""
        task1 = planner.create_task(
            task_id="task1",
            name="Task 1",
            description="Task 1"
        )
        task2 = planner.create_task(
            task_id="task2",
            name="Task 2",
            description="Task 2",
            dependencies=["task1"]
        )
        
        # Task2 should not be executable until task1 is completed
        assert not planner._can_execute_task(task2)
        
        task1.update_status(TaskStatus.COMPLETED)
        assert planner._can_execute_task(task2)
    
    def test_get_next_task(self, planner):
        """Test next task selection."""
        task1 = planner.create_task(
            task_id="task1",
            name="Task 1",
            description="Task 1",
            priority=1
        )
        task2 = planner.create_task(
            task_id="task2",
            name="Task 2",
            description="Task 2",
            priority=2
        )
        
        # Should select task2 due to higher priority
        next_task = planner._get_next_task()
        assert next_task == task2
    
    @pytest.mark.asyncio
    async def test_execute_task(self, planner):
        """Test task execution."""
        executed = False
        
        async def test_handler(context: TaskExecutionContext):
            nonlocal executed
            executed = True
        
        task = planner.create_task(
            task_id="test",
            name="Test Task",
            description="Test Task",
            metadata={"type": "test_type"}
        )
        
        planner.register_task_handler("test_type", test_handler)
        
        context = TaskExecutionContext(
            task=task,
            memory_manager=None
        )
        
        await planner.execute_task(task, context)
        assert executed
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_all_tasks(self, planner):
        """Test execution of all tasks."""
        executed_tasks = []
        
        async def test_handler(context: TaskExecutionContext):
            executed_tasks.append(context.task.id)
        
        # Create tasks with dependencies
        task1 = planner.create_task(
            task_id="task1",
            name="Task 1",
            description="Task 1",
            metadata={"type": "test_type"}
        )
        
        task2 = planner.create_task(
            task_id="task2",
            name="Task 2",
            description="Task 2",
            dependencies=["task1"],
            metadata={"type": "test_type"}
        )
        
        planner.register_task_handler("test_type", test_handler)
        
        await planner.execute_all_tasks(None)
        
        assert executed_tasks == ["task1", "task2"]
        assert all(
            task.status == TaskStatus.COMPLETED
            for task in planner.get_all_tasks()
        )
    
    def test_get_execution_plan(self, planner):
        """Test execution plan generation."""
        # Create tasks with dependencies
        task1 = planner.create_task(
            task_id="task1",
            name="Task 1",
            description="Task 1"
        )
        
        task2 = planner.create_task(
            task_id="task2",
            name="Task 2",
            description="Task 2",
            dependencies=["task1"]
        )
        
        task3 = planner.create_task(
            task_id="task3",
            name="Task 3",
            description="Task 3",
            dependencies=["task1"]
        )
        
        task4 = planner.create_task(
            task_id="task4",
            name="Task 4",
            description="Task 4",
            dependencies=["task2", "task3"]
        )
        
        plan = planner.get_execution_plan()
        
        # Verify the plan respects dependencies
        assert len(plan) == 3  # Should have 3 levels
        assert task1 in plan[0]  # First level should have task1
        assert all(t in plan[1] for t in [task2, task3])  # Second level should have task2 and task3
        assert task4 in plan[2]  # Third level should have task4
    
    @pytest.mark.asyncio
    async def test_task_failure_handling(self, planner):
        """Test handling of task failures."""
        async def failing_handler(context: TaskExecutionContext):
            raise ValueError("Test error")
        
        task1 = planner.create_task(
            task_id="task1",
            name="Task 1",
            description="Task 1",
            metadata={"type": "failing_type"}
        )
        
        task2 = planner.create_task(
            task_id="task2",
            name="Task 2",
            description="Task 2",
            dependencies=["task1"],
            metadata={"type": "failing_type"}
        )
        
        planner.register_task_handler("failing_type", failing_handler)
        
        with pytest.raises(Exception):
            await planner.execute_all_tasks(None)
        
        assert task1.status == TaskStatus.FAILED
        assert task2.status == TaskStatus.BLOCKED 
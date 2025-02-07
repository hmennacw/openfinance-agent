from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging

class TaskStatus(Enum):
    """Possible states for a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    """Represents a single task in the system."""
    id: str
    name: str
    description: str
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    subtasks: List['Task'] = field(default_factory=list)
    parent_id: Optional[str] = None
    priority: int = 0
    
    def add_subtask(self, subtask: 'Task') -> None:
        """Add a subtask to this task."""
        subtask.parent_id = self.id
        self.subtasks.append(subtask)
    
    def update_status(self, status: TaskStatus, error: Optional[str] = None) -> None:
        """Update the task's status."""
        self.status = status
        if status == TaskStatus.IN_PROGRESS and not self.started_at:
            self.started_at = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            self.completed_at = datetime.now()
        if error:
            self.error = error

@dataclass
class TaskExecutionContext:
    """Context for task execution."""
    task: Task
    memory_manager: Any  # Will be properly typed when memory manager is available
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable from the context."""
        return self.variables.get(key, default)
    
    def set_variable(self, key: str, value: Any) -> None:
        """Set a variable in the context."""
        self.variables[key] = value

class TaskPlanner:
    """Plans and manages task execution."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_task_handler(
        self,
        task_type: str,
        handler: Callable[[TaskExecutionContext], Any]
    ) -> None:
        """Register a handler for a specific task type."""
        self.handlers[task_type] = handler
    
    def create_task(
        self,
        task_id: str,
        name: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> Task:
        """Create a new task."""
        task = Task(
            id=task_id,
            name=name,
            description=description,
            dependencies=dependencies or [],
            metadata=metadata or {},
            priority=priority
        )
        self.tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks."""
        return list(self.tasks.values())
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.PENDING
        ]
    
    def get_blocked_tasks(self) -> List[Task]:
        """Get all blocked tasks."""
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.BLOCKED
        ]
    
    def _can_execute_task(self, task: Task) -> bool:
        """Check if a task can be executed."""
        if task.status != TaskStatus.PENDING:
            return False
        
        # Check if all dependencies are completed
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _get_next_task(self) -> Optional[Task]:
        """Get the next task to execute."""
        executable_tasks = [
            task for task in self.tasks.values()
            if self._can_execute_task(task)
        ]
        
        if not executable_tasks:
            return None
        
        # Return the highest priority task
        return max(executable_tasks, key=lambda t: t.priority)
    
    async def execute_task(
        self,
        task: Task,
        context: TaskExecutionContext
    ) -> None:
        """Execute a single task."""
        try:
            task.update_status(TaskStatus.IN_PROGRESS)
            
            # Get the appropriate handler for the task
            handler = self.handlers.get(task.metadata.get("type"))
            if not handler:
                raise ValueError(f"No handler found for task type: {task.metadata.get('type')}")
            
            # Execute the task
            await handler(context)
            
            # Update task status
            task.update_status(TaskStatus.COMPLETED)
            
        except Exception as e:
            self.logger.error(f"Error executing task {task.id}: {str(e)}")
            task.update_status(TaskStatus.FAILED, str(e))
            raise
    
    async def execute_all_tasks(
        self,
        memory_manager: Any
    ) -> None:
        """Execute all tasks in the correct order."""
        while True:
            next_task = self._get_next_task()
            if not next_task:
                # Check if there are any tasks that aren't completed or failed
                incomplete_tasks = [
                    task for task in self.tasks.values()
                    if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]
                ]
                if not incomplete_tasks:
                    break
                # If there are blocked tasks, we need to handle them
                blocked_tasks = self.get_blocked_tasks()
                if blocked_tasks:
                    self.logger.warning(f"Found {len(blocked_tasks)} blocked tasks")
                    # Here you could implement retry logic or other handling
                await asyncio.sleep(1)  # Prevent tight loop
                continue
            
            context = TaskExecutionContext(
                task=next_task,
                memory_manager=memory_manager
            )
            
            try:
                await self.execute_task(next_task, context)
            except Exception as e:
                self.logger.error(f"Failed to execute task {next_task.id}: {str(e)}")
                # Mark dependent tasks as blocked
                for task in self.tasks.values():
                    if next_task.id in task.dependencies:
                        task.update_status(TaskStatus.BLOCKED)
    
    def reset(self) -> None:
        """Reset the planner state."""
        self.tasks.clear()
        
    def get_execution_plan(self) -> List[List[Task]]:
        """Get the planned execution order of tasks."""
        # Implementation of topological sort for tasks
        visited = set()
        temp = set()
        order = []
        
        def visit(task: Task):
            if task.id in temp:
                raise ValueError("Circular dependency detected")
            if task.id in visited:
                return
            
            temp.add(task.id)
            
            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)
                if dep_task:
                    visit(dep_task)
            
            temp.remove(task.id)
            visited.add(task.id)
            order.append(task)
        
        # Sort tasks by visiting all nodes
        for task in self.tasks.values():
            if task.id not in visited:
                visit(task)
        
        # Group tasks that can be executed in parallel
        execution_plan = []
        current_level = []
        
        for task in reversed(order):
            if not task.dependencies or all(
                self.get_task(dep_id) in [t for level in execution_plan for t in level]
                for dep_id in task.dependencies
            ):
                current_level.append(task)
            else:
                if current_level:
                    execution_plan.append(current_level)
                current_level = [task]
        
        if current_level:
            execution_plan.append(current_level)
        
        return execution_plan 
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

@dataclass(frozen=True)
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
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return NotImplemented
        return self.id == other.id
    
    def add_subtask(self, subtask: 'Task') -> None:
        """Add a subtask to this task."""
        # Create a new task with the updated parent_id
        new_subtask = Task(
            id=subtask.id,
            name=subtask.name,
            description=subtask.description,
            dependencies=subtask.dependencies,
            status=subtask.status,
            created_at=subtask.created_at,
            started_at=subtask.started_at,
            completed_at=subtask.completed_at,
            error=subtask.error,
            metadata=subtask.metadata,
            subtasks=subtask.subtasks,
            parent_id=self.id,
            priority=subtask.priority
        )
        # Since the object is frozen, we need to use object.__setattr__ to modify the list
        object.__setattr__(self, 'subtasks', self.subtasks + [new_subtask])
    
    def update_status(self, status: TaskStatus, error: Optional[str] = None) -> None:
        """Update the task's status."""
        object.__setattr__(self, 'status', status)
        if status == TaskStatus.IN_PROGRESS and not self.started_at:
            object.__setattr__(self, 'started_at', datetime.now())
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            object.__setattr__(self, 'completed_at', datetime.now())
        if error:
            object.__setattr__(self, 'error', error)

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
        # First, create a mapping of tasks to their direct dependencies
        task_deps = {
            task: [self.get_task(dep_id) for dep_id in task.dependencies if self.get_task(dep_id)]
            for task in self.tasks.values()
        }
        
        # Create a mapping of tasks to their levels
        task_levels: Dict[Task, int] = {}
        
        def calculate_level(task: Task) -> int:
            """Calculate the level of a task based on its dependencies."""
            if task in task_levels:
                return task_levels[task]
            
            if not task_deps[task]:
                level = 0
            else:
                level = 1 + max(calculate_level(dep) for dep in task_deps[task])
            
            task_levels[task] = level
            return level
        
        # Calculate levels for all tasks
        for task in self.tasks.values():
            calculate_level(task)
        
        # Group tasks by level
        max_level = max(task_levels.values()) if task_levels else 0
        execution_plan = [[] for _ in range(max_level + 1)]
        
        for task, level in task_levels.items():
            execution_plan[level].append(task)
        
        return execution_plan 
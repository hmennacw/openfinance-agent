import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, Any

from src.cognitive.memory import MemoryManager
from src.cognitive.planner import TaskPlanner, Task, TaskStatus, TaskExecutionContext
from src.cognitive.decision import CodeGenerationDecisionMaker
from src.cognitive.learning import LearningSystem
from src.compiler.pipeline import (
    Pipeline,
    CompilationStage,
    SwaggerParser,
    CodeAnalyzer,
    CodeTransformer,
    CodeGenerator,
    CodeOptimizer,
    CodeValidator
)
from src.agent.llm.openai import OpenAIProvider
from src.agent.llm.prompts import PromptManager
from src.main import (
    setup_pipeline,
    setup_cognitive_system,
    setup_llm_provider,
    process_swagger_spec,
    main
)

@pytest.fixture
def mock_pipeline():
    pipeline = AsyncMock(spec=Pipeline)
    pipeline.add_stage = MagicMock()
    pipeline.set_context = MagicMock()
    pipeline.run = AsyncMock(return_value={"result": "success"})
    return pipeline

@pytest.fixture
def mock_memory_manager():
    manager = AsyncMock(spec=MemoryManager)
    manager.add_context_memory = MagicMock()
    manager.add_code_memory = MagicMock()
    manager.get_working_memory = MagicMock(return_value={
        "models": {"User": {"properties": {}}},
        "endpoints": [
            {
                "path": "/users",
                "method": "GET",
                "name": "list_users",
                "description": "List all users"
            }
        ]
    })
    return manager

@pytest.fixture
def mock_task_planner():
    planner = AsyncMock(spec=TaskPlanner)
    planner.create_task = MagicMock()
    planner.register_task_handler = MagicMock()
    planner.execute_all_tasks = AsyncMock()
    planner.get_all_tasks = MagicMock(return_value=[
        Task(
            id="parse_swagger",
            name="Parse Swagger",
            description="Parse swagger spec",
            dependencies=[],
            status=TaskStatus.COMPLETED
        ),
        Task(
            id="generate_models",
            name="Generate Models",
            description="Generate models",
            dependencies=[],
            status=TaskStatus.FAILED,
            error="Test error"
        )
    ])
    return planner

@pytest.fixture
def mock_decision_maker():
    return AsyncMock(spec=CodeGenerationDecisionMaker)

@pytest.fixture
def mock_learning_system():
    system = AsyncMock(spec=LearningSystem)
    system.add_example = MagicMock()
    return system

@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock(spec=OpenAIProvider)
    provider.initialize = AsyncMock()
    provider.generate_completion = AsyncMock(return_value="Generated code")
    return provider

@pytest.mark.asyncio
async def test_setup_pipeline():
    """Test pipeline setup."""
    pipeline = await setup_pipeline()
    assert isinstance(pipeline, Pipeline)
    
    # Verify all stages are added
    stage_types = [
        (CompilationStage.PARSING, SwaggerParser),
        (CompilationStage.ANALYSIS, CodeAnalyzer),
        (CompilationStage.TRANSFORMATION, CodeTransformer),
        (CompilationStage.GENERATION, CodeGenerator),
        (CompilationStage.OPTIMIZATION, CodeOptimizer),
        (CompilationStage.VALIDATION, CodeValidator)
    ]
    
    for stage, stage_type in stage_types:
        stage_processors = pipeline.stages.get(stage)
        assert len(stage_processors) == 1
        assert isinstance(stage_processors[0], stage_type)

@pytest.mark.asyncio
async def test_setup_cognitive_system():
    """Test cognitive system setup."""
    storage_path = Path("/tmp/test")
    
    memory_manager, task_planner, decision_maker, learning_system = (
        await setup_cognitive_system(storage_path)
    )
    
    assert isinstance(memory_manager, MemoryManager)
    assert isinstance(task_planner, TaskPlanner)
    assert isinstance(decision_maker, CodeGenerationDecisionMaker)
    assert isinstance(learning_system, LearningSystem)
    assert learning_system.storage_path == str(storage_path / "learning_data.json")

@pytest.mark.asyncio
async def test_setup_llm_provider():
    """Test LLM provider setup."""
    provider = await setup_llm_provider()
    assert isinstance(provider, OpenAIProvider)

@pytest.mark.asyncio
async def test_process_swagger_spec(
    mock_pipeline,
    mock_memory_manager,
    mock_task_planner,
    mock_decision_maker,
    mock_learning_system,
    mock_llm_provider,
    tmp_path: Path
):
    """Test Swagger spec processing."""
    # Create test swagger file
    swagger_path = tmp_path / "api.yaml"
    swagger_path.write_text("swagger: '2.0'")
    
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    
    await process_swagger_spec(
        swagger_path=swagger_path,
        output_dir=output_dir,
        llm_provider=mock_llm_provider,
        pipeline=mock_pipeline,
        memory_manager=mock_memory_manager,
        task_planner=mock_task_planner,
        decision_maker=mock_decision_maker,
        learning_system=mock_learning_system
    )
    
    # Verify task creation
    expected_tasks = [
        {
            "id": "parse_swagger",
            "name": "Parse Swagger Specification",
            "description": "Parse and validate the Swagger specification",
            "priority": 1,
            "dependencies": []
        },
        {
            "id": "analyze_endpoints",
            "name": "Analyze API Endpoints",
            "description": "Analyze the API endpoints and determine code structure",
            "priority": 1,
            "dependencies": ["parse_swagger"]
        },
        {
            "id": "generate_models",
            "name": "Generate Data Models",
            "description": "Generate Go structs for data models",
            "priority": 2,
            "dependencies": ["analyze_endpoints"]
        },
        {
            "id": "generate_handlers",
            "name": "Generate API Handlers",
            "description": "Generate Go handlers for API endpoints",
            "priority": 2,
            "dependencies": ["generate_models"]
        },
        {
            "id": "generate_routes",
            "name": "Generate API Routes",
            "description": "Generate Go route configurations",
            "priority": 2,
            "dependencies": ["generate_handlers"]
        }
    ]
    
    # Verify each task was created with the correct parameters
    assert mock_task_planner.create_task.call_count == len(expected_tasks)
    for task in expected_tasks:
        mock_task_planner.create_task.assert_any_call(**task)
    
    # Verify task handlers are registered
    assert mock_task_planner.register_task_handler.call_count == 5
    
    # Verify tasks are executed
    assert mock_task_planner.execute_all_tasks.called
    mock_task_planner.execute_all_tasks.assert_called_once_with(mock_memory_manager)
    
    # Verify learning examples are added
    assert mock_learning_system.add_example.call_count == 2
    mock_learning_system.add_example.assert_has_calls([
        call(
            example_id="parse_swagger",
            context={"task_type": "parse_swagger"},
            decision={"chosen_option": "success"},
            outcome={"status": "completed"},
            tags=["parse_swagger", "success"]
        ),
        call(
            example_id="generate_models",
            context={"task_type": "generate_models"},
            decision={"chosen_option": "failure"},
            outcome={"status": "failed", "error": "Test error"},
            tags=["generate_models", "failure"]
        )
    ])

@pytest.mark.asyncio
async def test_main(
    mock_pipeline,
    mock_memory_manager,
    mock_task_planner,
    mock_decision_maker,
    mock_learning_system,
    mock_llm_provider,
    tmp_path: Path
):
    """Test main function."""
    # Mock the setup functions
    with patch("src.main.setup_pipeline", return_value=mock_pipeline), \
         patch("src.main.setup_cognitive_system", return_value=(
             mock_memory_manager,
             mock_task_planner,
             mock_decision_maker,
             mock_learning_system
         )), \
         patch("src.main.setup_llm_provider", return_value=mock_llm_provider), \
         patch("src.main.process_swagger_spec") as mock_process:
        
        # Create test directories and files
        base_path = tmp_path / "project"
        base_path.mkdir()
        examples_path = base_path / "examples"
        examples_path.mkdir()
        swagger_path = examples_path / "swagger"
        swagger_path.mkdir()
        (swagger_path / "api.yaml").write_text("swagger: '2.0'")
        
        # Mock Path.parent.parent to return our test base_path
        with patch("pathlib.Path.parent", new_callable=MagicMock) as mock_parent:
            mock_parent.parent = base_path
            
            await main()
            
            # Verify directories are created
            assert (base_path / "examples" / "generated").exists()
            assert (base_path / "data").exists()
            
            # Verify process_swagger_spec is called
            assert mock_process.called
            mock_process.assert_called_once()

@pytest.mark.asyncio
async def test_task_handlers(
    mock_pipeline,
    mock_memory_manager,
    mock_task_planner,
    mock_decision_maker,
    mock_learning_system,
    mock_llm_provider,
    tmp_path: Path
):
    """Test individual task handlers."""
    swagger_path = tmp_path / "api.yaml"
    swagger_path.write_text("swagger: '2.0'")
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    
    # Create a real PromptManager instance
    prompt_manager = PromptManager()
    
    # Process the swagger spec to get access to task handlers
    await process_swagger_spec(
        swagger_path=swagger_path,
        output_dir=output_dir,
        llm_provider=mock_llm_provider,
        pipeline=mock_pipeline,
        memory_manager=mock_memory_manager,
        task_planner=mock_task_planner,
        decision_maker=mock_decision_maker,
        learning_system=mock_learning_system
    )
    
    # Get the registered handlers
    handlers = {
        call[0][0]: call[0][1]
        for call in mock_task_planner.register_task_handler.call_args_list
    }
    
    # Test parse_swagger handler
    await handlers["parse_swagger"]({})
    mock_pipeline.set_context.assert_called_with("swagger_spec", "swagger: '2.0'")
    mock_pipeline.run.assert_called_with({"stage": "parsing"})
    mock_memory_manager.add_context_memory.assert_called_with(
        context_type="parsing",
        scope="swagger",
        content={"result": "success"}
    )
    
    # Test analyze_endpoints handler
    await handlers["analyze_endpoints"]({})
    mock_pipeline.run.assert_called_with({
        "stage": "analysis",
        "parsed_spec": mock_memory_manager.get_working_memory.return_value
    })
    
    # Test generate_models handler
    await handlers["generate_models"]({})
    mock_llm_provider.generate_completion.assert_called()
    mock_memory_manager.add_code_memory.assert_called_with(
        file_path=str(output_dir / "models.go"),
        code_type="model",
        content="Generated code"
    )
    
    # Test generate_handlers handler
    await handlers["generate_handlers"]({})
    assert mock_llm_provider.generate_completion.call_count > 0
    mock_memory_manager.add_code_memory.assert_called_with(
        file_path=str(output_dir / "handler_list_users.go"),
        code_type="handler",
        content="Generated code"
    )
    
    # Test generate_routes handler
    await handlers["generate_routes"]({})
    mock_llm_provider.generate_completion.assert_called()
    mock_memory_manager.add_code_memory.assert_called_with(
        file_path=str(output_dir / "router.go"),
        code_type="router",
        content="Generated code"
    )

@pytest.mark.asyncio
async def test_execute_all_tasks_with_failure(
    mock_task_planner,
    mock_memory_manager
):
    """Test execution with task failure."""
    async def failing_handler(context: TaskExecutionContext):
        raise ValueError("Test error")
    
    task1 = mock_task_planner.create_task(
        id="task1",
        name="Task 1",
        description="First task",
        metadata={"type": "failing_type"}
    )
    
    task2 = mock_task_planner.create_task(
        id="task2",
        name="Task 2",
        description="Second task",
        dependencies=["task1"],
        metadata={"type": "failing_type"}
    )

    def test_get_execution_plan(mock_task_planner):
        """Test execution plan generation."""
        # Create tasks with dependencies
        task1 = mock_task_planner.create_task(
            id="task1",
            name="Task 1",
            description="First task"
        )
        
        task2 = mock_task_planner.create_task(
            id="task2",
            name="Task 2",
            description="Second task",
            dependencies=["task1"]
        )
        
        task3 = mock_task_planner.create_task(
            id="task3",
            name="Task 3",
            description="Third task",
            dependencies=["task1"]
        )
        
        task4 = mock_task_planner.create_task(
            id="task4",
            name="Task 4",
            description="Fourth task",
            dependencies=["task2", "task3"]
        )

    @pytest.mark.asyncio
    async def test_execute_all_tasks_with_blocked(
        mock_task_planner,
        mock_memory_manager
    ):
        """Test execution with blocked tasks."""
        async def test_handler(context: TaskExecutionContext):
            pass
        
        # Create tasks with missing dependency
        task = mock_task_planner.create_task(
            id="task",
            name="Task",
            description="Task with missing dependency",
            dependencies=["missing"],
            metadata={"type": "test_type"}
        ) 
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from cognitive.memory import MemoryManager
from cognitive.planner import TaskPlanner, Task, TaskStatus
from cognitive.decision import CodeGenerationDecisionMaker
from cognitive.learning import LearningSystem
from compiler.pipeline import (
    Pipeline,
    CompilationStage,
    SwaggerParser,
    CodeAnalyzer,
    CodeTransformer,
    CodeGenerator,
    CodeOptimizer,
    CodeValidator
)
from agent.llm.openai import OpenAIProvider
from agent.llm.prompts import PromptManager

async def setup_pipeline() -> Pipeline:
    """Set up the compilation pipeline."""
    pipeline = Pipeline()
    
    # Add pipeline stages
    pipeline.add_stage(CompilationStage.PARSING, SwaggerParser())
    pipeline.add_stage(CompilationStage.ANALYSIS, CodeAnalyzer())
    pipeline.add_stage(CompilationStage.TRANSFORMATION, CodeTransformer())
    pipeline.add_stage(CompilationStage.GENERATION, CodeGenerator())
    pipeline.add_stage(CompilationStage.OPTIMIZATION, CodeOptimizer())
    pipeline.add_stage(CompilationStage.VALIDATION, CodeValidator())
    
    return pipeline

async def setup_cognitive_system(
    storage_path: Path
) -> tuple[MemoryManager, TaskPlanner, CodeGenerationDecisionMaker, LearningSystem]:
    """Set up the cognitive architecture components."""
    memory_manager = MemoryManager()
    task_planner = TaskPlanner()
    decision_maker = CodeGenerationDecisionMaker()
    learning_system = LearningSystem(
        storage_path=str(storage_path / "learning_data.json")
    )
    
    return memory_manager, task_planner, decision_maker, learning_system

async def setup_llm_provider() -> OpenAIProvider:
    """Set up the LLM provider."""
    provider = OpenAIProvider()
    await provider.initialize()
    return provider

async def process_swagger_spec(
    swagger_path: Path,
    output_dir: Path,
    llm_provider: OpenAIProvider,
    pipeline: Pipeline,
    memory_manager: MemoryManager,
    task_planner: TaskPlanner,
    decision_maker: CodeGenerationDecisionMaker,
    learning_system: LearningSystem
) -> None:
    """Process a Swagger specification and generate Go code."""
    # Load and parse the Swagger spec
    with open(swagger_path) as f:
        swagger_spec = f.read()
    
    # Create the prompt manager
    prompt_manager = PromptManager()
    
    # Create initial tasks
    task_planner.create_task(
        task_id="parse_swagger",
        name="Parse Swagger Specification",
        description="Parse and validate the Swagger specification",
        priority=1
    )
    
    task_planner.create_task(
        task_id="analyze_endpoints",
        name="Analyze API Endpoints",
        description="Analyze the API endpoints and determine code structure",
        dependencies=["parse_swagger"],
        priority=1
    )
    
    task_planner.create_task(
        task_id="generate_models",
        name="Generate Data Models",
        description="Generate Go structs for data models",
        dependencies=["analyze_endpoints"],
        priority=2
    )
    
    task_planner.create_task(
        task_id="generate_handlers",
        name="Generate API Handlers",
        description="Generate Go handlers for API endpoints",
        dependencies=["generate_models"],
        priority=2
    )
    
    task_planner.create_task(
        task_id="generate_routes",
        name="Generate API Routes",
        description="Generate Go route configurations",
        dependencies=["generate_handlers"],
        priority=2
    )
    
    # Register task handlers
    async def handle_parse_swagger(context: Dict[str, Any]) -> None:
        pipeline.set_context("swagger_spec", swagger_spec)
        result = await pipeline.run({"stage": "parsing"})
        memory_manager.add_context_memory(
            context_type="parsing",
            scope="swagger",
            content=result
        )
    
    async def handle_analyze_endpoints(context: Dict[str, Any]) -> None:
        parsed_spec = memory_manager.get_working_memory("parsed_spec")
        result = await pipeline.run({
            "stage": "analysis",
            "parsed_spec": parsed_spec
        })
        memory_manager.add_context_memory(
            context_type="analysis",
            scope="endpoints",
            content=result
        )
    
    async def handle_generate_models(context: Dict[str, Any]) -> None:
        analysis = memory_manager.get_working_memory("analysis")
        prompt = prompt_manager.generate_model_prompt(
            schema=analysis["models"]
        )
        response = await llm_provider.generate_completion(prompt)
        memory_manager.add_code_memory(
            file_path=str(output_dir / "models.go"),
            code_type="model",
            content=response
        )
    
    async def handle_generate_handlers(context: Dict[str, Any]) -> None:
        analysis = memory_manager.get_working_memory("analysis")
        for endpoint in analysis["endpoints"]:
            prompt = prompt_manager.generate_handler_prompt(
                path=endpoint["path"],
                method=endpoint["method"],
                description=endpoint["description"]
            )
            response = await llm_provider.generate_completion(prompt)
            memory_manager.add_code_memory(
                file_path=str(output_dir / f"handler_{endpoint['name']}.go"),
                code_type="handler",
                content=response
            )
    
    async def handle_generate_routes(context: Dict[str, Any]) -> None:
        analysis = memory_manager.get_working_memory("analysis")
        prompt = prompt_manager.generate_usecase_prompt(
            name="router",
            description="Configure API routes"
        )
        response = await llm_provider.generate_completion(prompt)
        memory_manager.add_code_memory(
            file_path=str(output_dir / "router.go"),
            code_type="router",
            content=response
        )
    
    task_planner.register_task_handler("parse_swagger", handle_parse_swagger)
    task_planner.register_task_handler("analyze_endpoints", handle_analyze_endpoints)
    task_planner.register_task_handler("generate_models", handle_generate_models)
    task_planner.register_task_handler("generate_handlers", handle_generate_handlers)
    task_planner.register_task_handler("generate_routes", handle_generate_routes)
    
    # Execute all tasks
    await task_planner.execute_all_tasks(memory_manager)
    
    # Save learning examples
    for task in task_planner.get_all_tasks():
        if task.status == TaskStatus.COMPLETED:
            learning_system.add_example(
                example_id=task.id,
                context={"task_type": task.id},
                decision={"chosen_option": "success"},
                outcome={"status": "completed"},
                tags=[task.id, "success"]
            )
        elif task.status == TaskStatus.FAILED:
            learning_system.add_example(
                example_id=task.id,
                context={"task_type": task.id},
                decision={"chosen_option": "failure"},
                outcome={"status": "failed", "error": task.error},
                tags=[task.id, "failure"]
            )

async def main():
    """Main entry point."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up paths
    base_path = Path(__file__).parent.parent
    swagger_path = base_path / "examples" / "swagger" / "api.yaml"
    output_dir = base_path / "examples" / "generated"
    storage_path = base_path / "data"
    
    # Create necessary directories
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Set up components
    pipeline = await setup_pipeline()
    memory_manager, task_planner, decision_maker, learning_system = (
        await setup_cognitive_system(storage_path)
    )
    llm_provider = await setup_llm_provider()
    
    # Process the Swagger specification
    await process_swagger_spec(
        swagger_path=swagger_path,
        output_dir=output_dir,
        llm_provider=llm_provider,
        pipeline=pipeline,
        memory_manager=memory_manager,
        task_planner=task_planner,
        decision_maker=decision_maker,
        learning_system=learning_system
    )

if __name__ == "__main__":
    asyncio.run(main()) 
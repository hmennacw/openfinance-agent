import pytest
from typing import Dict, Any
import logging

from src.compiler.pipeline import (
    CompilationStage,
    CompilationError,
    Pipeline,
    PipelineStage,
    SwaggerParser,
    CodeAnalyzer,
    CodeTransformer,
    CodeGenerator,
    CodeOptimizer,
    CodeValidator
)

class MockPipelineStage:
    """Mock pipeline stage for testing."""
    
    def __init__(self, stage_name: str, should_fail: bool = False):
        self.stage_name = stage_name
        self.should_fail = should_fail
        self.was_called = False
        self.context = None
    
    async def process(self, context: Dict[str, Any]) -> None:
        """Process the context."""
        self.was_called = True
        self.context = context
        if self.should_fail:
            raise ValueError(f"Mock error in {self.stage_name}")

def test_compilation_stage_enum():
    """Test compilation stage enumeration."""
    assert CompilationStage.PARSING.value == "parsing"
    assert CompilationStage.ANALYSIS.value == "analysis"
    assert CompilationStage.TRANSFORMATION.value == "transformation"
    assert CompilationStage.GENERATION.value == "generation"
    assert CompilationStage.OPTIMIZATION.value == "optimization"
    assert CompilationStage.VALIDATION.value == "validation"

def test_compilation_error():
    """Test compilation error creation."""
    error = CompilationError("Test error", CompilationStage.PARSING)
    assert str(error) == "parsing: Test error"
    assert error.stage == CompilationStage.PARSING

class TestPipeline:
    """Test suite for Pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance."""
        return Pipeline()
    
    def test_add_stage(self, pipeline):
        """Test adding a stage to the pipeline."""
        stage = MockPipelineStage("test_stage")
        pipeline.add_stage(CompilationStage.PARSING, stage)
        assert stage in pipeline.stages[CompilationStage.PARSING]
    
    def test_context_management(self, pipeline):
        """Test context management."""
        pipeline.set_context("key", "value")
        assert pipeline.get_context("key") == "value"
        assert pipeline.get_context("nonexistent", "default") == "default"
    
    @pytest.mark.asyncio
    async def test_run_pipeline(self, pipeline):
        """Test running the complete pipeline."""
        stages = {
            CompilationStage.PARSING: MockPipelineStage("parsing"),
            CompilationStage.ANALYSIS: MockPipelineStage("analysis"),
            CompilationStage.TRANSFORMATION: MockPipelineStage("transformation")
        }
        
        for stage_type, stage in stages.items():
            pipeline.add_stage(stage_type, stage)
        
        input_data = {"test": "data"}
        result = await pipeline.run(input_data)
        
        assert result["test"] == "data"
        for stage in stages.values():
            assert stage.was_called
            assert stage.context["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, pipeline):
        """Test pipeline error handling."""
        failing_stage = MockPipelineStage("failing", should_fail=True)
        pipeline.add_stage(CompilationStage.PARSING, failing_stage)
        
        with pytest.raises(CompilationError) as exc_info:
            await pipeline.run({})
        
        assert exc_info.value.stage == CompilationStage.PARSING
        assert "Mock error" in str(exc_info.value)

class TestSwaggerParser:
    """Test suite for SwaggerParser."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return SwaggerParser()
    
    @pytest.mark.asyncio
    async def test_parse_swagger(self, parser, sample_swagger_spec):
        """Test parsing Swagger specification."""
        context = {"swagger_spec": sample_swagger_spec}
        await parser.process(context)
        
        assert "parsed_spec" in context
        parsed = context["parsed_spec"]
        assert parsed["openapi"] == "3.0.0"
        assert parsed["info"]["title"] == "Test API"
    
    @pytest.mark.asyncio
    async def test_parse_swagger_missing_spec(self, parser):
        """Test parsing with missing specification."""
        with pytest.raises(CompilationError) as exc_info:
            await parser.process({})
        
        assert exc_info.value.stage == CompilationStage.PARSING
        assert "No Swagger specification provided" in str(exc_info.value)

class TestCodeAnalyzer:
    """Test suite for CodeAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance."""
        return CodeAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_spec(self, analyzer):
        """Test analyzing specification."""
        context = {
            "parsed_spec": {
                "paths": {
                    "/test": {
                        "get": {
                            "summary": "Test endpoint"
                        }
                    }
                }
            }
        }
        
        await analyzer.process(context)
        assert "analysis" in context
        analysis = context["analysis"]
        assert "endpoints" in analysis
        assert "models" in analysis
        assert "dependencies" in analysis
    
    @pytest.mark.asyncio
    async def test_analyze_missing_spec(self, analyzer):
        """Test analyzing with missing specification."""
        with pytest.raises(CompilationError) as exc_info:
            await analyzer.process({})
        
        assert exc_info.value.stage == CompilationStage.ANALYSIS
        assert "No parsed specification available" in str(exc_info.value)

class TestCodeTransformer:
    """Test suite for CodeTransformer."""
    
    @pytest.fixture
    def transformer(self):
        """Create a transformer instance."""
        return CodeTransformer()
    
    @pytest.mark.asyncio
    async def test_transform_analysis(self, transformer):
        """Test transforming analysis results."""
        context = {
            "analysis": {
                "endpoints": [],
                "models": [],
                "dependencies": []
            }
        }
        
        await transformer.process(context)
        assert "ir" in context
        ir = context["ir"]
        assert "handlers" in ir
        assert "models" in ir
        assert "routes" in ir
    
    @pytest.mark.asyncio
    async def test_transform_missing_analysis(self, transformer):
        """Test transforming with missing analysis."""
        with pytest.raises(CompilationError) as exc_info:
            await transformer.process({})
        
        assert exc_info.value.stage == CompilationStage.TRANSFORMATION
        assert "No analysis results available" in str(exc_info.value)

class TestCodeGenerator:
    """Test suite for CodeGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return CodeGenerator()
    
    @pytest.mark.asyncio
    async def test_generate_code(self, generator):
        """Test generating code."""
        context = {
            "ir": {
                "handlers": [],
                "models": [],
                "routes": []
            }
        }
        
        await generator.process(context)
        assert "generated_code" in context
        code = context["generated_code"]
        assert "handlers" in code
        assert "models" in code
        assert "routes" in code
    
    @pytest.mark.asyncio
    async def test_generate_missing_ir(self, generator):
        """Test generating with missing IR."""
        with pytest.raises(CompilationError) as exc_info:
            await generator.process({})
        
        assert exc_info.value.stage == CompilationStage.GENERATION
        assert "No intermediate representation available" in str(exc_info.value)

class TestCodeOptimizer:
    """Test suite for CodeOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create an optimizer instance."""
        return CodeOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimize_code(self, optimizer):
        """Test optimizing code."""
        context = {
            "generated_code": {
                "test.go": "package test\n\nfunc main() {}\n"
            }
        }
        
        await optimizer.process(context)
        assert "optimized_code" in context
        assert "test.go" in context["optimized_code"]
    
    @pytest.mark.asyncio
    async def test_optimize_missing_code(self, optimizer):
        """Test optimizing with missing code."""
        with pytest.raises(CompilationError) as exc_info:
            await optimizer.process({})
        
        assert exc_info.value.stage == CompilationStage.OPTIMIZATION
        assert "No generated code available" in str(exc_info.value)

class TestCodeValidator:
    """Test suite for CodeValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return CodeValidator()
    
    @pytest.mark.asyncio
    async def test_validate_code(self, validator):
        """Test validating code."""
        context = {
            "optimized_code": {
                "test.go": "package test\n\nfunc main() {}\n"
            }
        }
        
        await validator.process(context)
        assert "validation_results" in context
        results = context["validation_results"]
        assert results["valid"]
        assert not results["errors"]
    
    @pytest.mark.asyncio
    async def test_validate_missing_code(self, validator):
        """Test validating with missing code."""
        with pytest.raises(CompilationError) as exc_info:
            await validator.process({})
        
        assert exc_info.value.stage == CompilationStage.VALIDATION
        assert "No optimized code available" in str(exc_info.value) 
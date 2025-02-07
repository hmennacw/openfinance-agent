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

@pytest.fixture
def sample_spec() -> Dict[str, Any]:
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0"
        },
        "paths": {
            "/users": {
                "post": {
                    "summary": "Create user",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "email": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "User created"
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                }
            }
        }
    }

class TestCodeAnalyzer:
    """Test suite for CodeAnalyzer."""
    
    @pytest.fixture
    def analyzer(self) -> CodeAnalyzer:
        """Create a CodeAnalyzer instance."""
        return CodeAnalyzer()
    
    @pytest.mark.asyncio
    async def test_process_without_spec(self, analyzer: CodeAnalyzer):
        """Test process without a parsed specification."""
        context: Dict[str, Any] = {}
        with pytest.raises(CompilationError) as exc_info:
            await analyzer.process(context)
        
        assert exc_info.value.stage == CompilationStage.ANALYSIS
        assert "No parsed specification available" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_with_spec(
        self,
        analyzer: CodeAnalyzer,
        sample_spec: Dict[str, Any]
    ):
        """Test process with a valid specification."""
        context = {"parsed_spec": sample_spec}
        await analyzer.process(context)
        
        analysis = context.get("analysis")
        assert analysis is not None
        assert "endpoints" in analysis
        assert "models" in analysis
        assert "dependencies" in analysis
    
    def test_analyze_endpoints(self, analyzer: CodeAnalyzer, sample_spec: Dict[str, Any]):
        """Test endpoint analysis."""
        endpoints = analyzer._analyze_endpoints(sample_spec)
        assert isinstance(endpoints, list)
        assert len(endpoints) == 1  # One endpoint in sample spec
        
        endpoint = endpoints[0]
        assert endpoint["path"] == "/users"
        assert endpoint["method"] == "post"
        assert endpoint["summary"] == "Create user"
        assert "requestBody" in endpoint
        assert "responses" in endpoint
        assert "201" in endpoint["responses"]
    
    def test_analyze_models(self, analyzer: CodeAnalyzer, sample_spec: Dict[str, Any]):
        """Test model analysis."""
        models = analyzer._analyze_models(sample_spec)
        assert isinstance(models, list)
        assert len(models) == 1  # One model in sample spec
        
        model = models[0]
        assert model["name"] == "User"
        assert model["type"] == "object"
        assert len(model["properties"]) == 3
        assert all(prop in model["properties"] for prop in ["id", "name", "email"])
        assert model["properties"]["id"]["type"] == "integer"
        assert model["properties"]["name"]["type"] == "string"
        assert model["properties"]["email"]["type"] == "string"
    
    def test_analyze_dependencies(self, analyzer: CodeAnalyzer, sample_spec: Dict[str, Any]):
        """Test dependency analysis."""
        dependencies = analyzer._analyze_dependencies(sample_spec)
        assert isinstance(dependencies, list)
        # Should include at least basic Go dependencies
        assert "net/http" in dependencies
        assert "encoding/json" in dependencies
        
    def test_analyze_empty_spec(self, analyzer: CodeAnalyzer):
        """Test analyzing an empty specification."""
        empty_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Empty API", "version": "1.0.0"},
            "paths": {},
            "components": {"schemas": {}}
        }
        
        analysis = analyzer._analyze_spec(empty_spec)
        assert analysis["endpoints"] == []
        assert analysis["models"] == []
        assert isinstance(analysis["dependencies"], list)  # Should still have basic dependencies
        
    def test_analyze_complex_types(self, analyzer: CodeAnalyzer):
        """Test analyzing complex data types."""
        spec = {
            "components": {
                "schemas": {
                    "ComplexType": {
                        "type": "object",
                        "properties": {
                            "array_field": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "nested_object": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "integer"}
                                }
                            },
                            "enum_field": {
                                "type": "string",
                                "enum": ["A", "B", "C"]
                            }
                        }
                    }
                }
            }
        }
        
        models = analyzer._analyze_models(spec)
        assert len(models) == 1
        model = models[0]
        assert model["name"] == "ComplexType"
        assert "array_field" in model["properties"]
        assert "nested_object" in model["properties"]
        assert "enum_field" in model["properties"]
        assert model["properties"]["enum_field"]["enum"] == ["A", "B", "C"]

class TestCodeTransformer:
    """Test suite for CodeTransformer."""
    
    @pytest.fixture
    def transformer(self):
        """Create a transformer instance."""
        return CodeTransformer()
    
    @pytest.fixture
    def sample_analysis(self) -> Dict[str, Any]:
        """Sample analysis results."""
        return {
            "endpoints": [{
                "path": "/users",
                "method": "post",
                "summary": "Create user",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/User"}
                        }
                    }
                },
                "responses": {
                    "201": {"description": "User created"}
                }
            }],
            "models": [{
                "name": "User",
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                }
            }],
            "dependencies": ["net/http", "encoding/json"]
        }
    
    @pytest.mark.asyncio
    async def test_transform_analysis(
        self,
        transformer: CodeTransformer,
        sample_analysis: Dict[str, Any]
    ):
        """Test transforming analysis results."""
        context = {"analysis": sample_analysis}
        await transformer.process(context)
        
        assert "ir" in context
        ir = context["ir"]
        
        # Check handlers
        assert "handlers" in ir
        handlers = ir["handlers"]
        assert len(handlers) == 1
        handler = handlers[0]
        assert handler["path"] == "/users"
        assert handler["method"] == "post"
        assert "requestType" in handler
        assert "responseType" in handler
        
        # Check models
        assert "models" in ir
        models = ir["models"]
        assert len(models) == 1
        model = models[0]
        assert model["name"] == "User"
        assert len(model["fields"]) == 3
        assert all(field["name"] in ["id", "name", "email"] for field in model["fields"])
        
        # Check routes
        assert "routes" in ir
        routes = ir["routes"]
        assert len(routes) == 1
        route = routes[0]
        assert route["path"] == "/users"
        assert route["method"] == "post"
        assert route["handler"] == "CreateUserHandler"
    
    @pytest.mark.asyncio
    async def test_transform_missing_analysis(self, transformer: CodeTransformer):
        """Test transforming with missing analysis."""
        with pytest.raises(CompilationError) as exc_info:
            await transformer.process({})
        
        assert exc_info.value.stage == CompilationStage.TRANSFORMATION
        assert "No analysis results available" in str(exc_info.value)
    
    def test_transform_handlers(self, transformer: CodeTransformer, sample_analysis: Dict[str, Any]):
        """Test handler transformation."""
        handlers = transformer._transform_handlers(sample_analysis)
        assert len(handlers) == 1
        
        handler = handlers[0]
        assert handler["name"] == "CreateUserHandler"
        assert handler["method"] == "post"
        assert handler["path"] == "/users"
        assert "requestValidation" in handler
        assert "errorHandling" in handler
    
    def test_transform_models(self, transformer: CodeTransformer, sample_analysis: Dict[str, Any]):
        """Test model transformation."""
        models = transformer._transform_models(sample_analysis)
        assert len(models) == 1
        
        model = models[0]
        assert model["name"] == "User"
        assert model["package"] == "models"
        assert len(model["fields"]) == 3
        
        # Check field types are correctly mapped to Go types
        field_types = {field["name"]: field["type"] for field in model["fields"]}
        assert field_types["id"] == "int64"
        assert field_types["name"] == "string"
        assert field_types["email"] == "string"
    
    def test_transform_routes(self, transformer: CodeTransformer, sample_analysis: Dict[str, Any]):
        """Test route transformation."""
        routes = transformer._transform_routes(sample_analysis)
        assert len(routes) == 1
        
        route = routes[0]
        assert route["path"] == "/users"
        assert route["method"] == "post"
        assert route["handler"] == "CreateUserHandler"
        assert route["middleware"] == []  # No middleware specified
    
    def test_transform_empty_analysis(self, transformer: CodeTransformer):
        """Test transforming empty analysis."""
        empty_analysis = {
            "endpoints": [],
            "models": [],
            "dependencies": []
        }
        
        ir = transformer._transform(empty_analysis)
        assert ir["handlers"] == []
        assert ir["models"] == []
        assert ir["routes"] == []

class TestCodeGenerator:
    """Test suite for CodeGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return CodeGenerator()
    
    @pytest.fixture
    def sample_ir(self) -> Dict[str, Any]:
        """Sample intermediate representation."""
        return {
            "handlers": [{
                "name": "CreateUserHandler",
                "method": "post",
                "path": "/users",
                "requestType": "CreateUserRequest",
                "responseType": "User",
                "requestValidation": True,
                "errorHandling": True
            }],
            "models": [{
                "name": "User",
                "package": "models",
                "fields": [
                    {"name": "id", "type": "int64", "tags": ['json:"id"']},
                    {"name": "name", "type": "string", "tags": ['json:"name"']},
                    {"name": "email", "type": "string", "tags": ['json:"email"']}
                ]
            }],
            "routes": [{
                "path": "/users",
                "method": "post",
                "handler": "CreateUserHandler",
                "middleware": []
            }]
        }
    
    @pytest.mark.asyncio
    async def test_generate_code(
        self,
        generator: CodeGenerator,
        sample_ir: Dict[str, Any]
    ):
        """Test generating code."""
        context = {"ir": sample_ir}
        await generator.process(context)
        
        assert "generated_code" in context
        code = context["generated_code"]
        
        # Check handler code
        assert "handlers/user_handler.go" in code
        handler_code = code["handlers/user_handler.go"]
        assert "package handlers" in handler_code
        assert "type CreateUserHandler struct" in handler_code
        assert "func (h *CreateUserHandler) Handle" in handler_code
        
        # Check model code
        assert "models/user.go" in code
        model_code = code["models/user.go"]
        assert "package models" in model_code
        assert "type User struct" in model_code
        assert 'json:"id"' in model_code
        assert 'json:"name"' in model_code
        assert 'json:"email"' in model_code
        
        # Check router code
        assert "api/router.go" in code
        router_code = code["api/router.go"]
        assert "package api" in router_code
        assert "func SetupRoutes" in router_code
        assert '"/users"' in router_code
    
    @pytest.mark.asyncio
    async def test_generate_missing_ir(self, generator: CodeGenerator):
        """Test generating with missing IR."""
        with pytest.raises(CompilationError) as exc_info:
            await generator.process({})
        
        assert exc_info.value.stage == CompilationStage.GENERATION
        assert "No intermediate representation available" in str(exc_info.value)
    
    def test_generate_handler_code(self, generator: CodeGenerator, sample_ir: Dict[str, Any]):
        """Test handler code generation."""
        handler = sample_ir["handlers"][0]
        code = generator._generate_handler_code(handler)
        
        # Check imports
        assert "net/http" in code
        assert "encoding/json" in code
        
        # Check struct definition
        assert "type CreateUserHandler struct" in code
        
        # Check method implementation
        assert "func (h *CreateUserHandler) Handle" in code
        assert "var request CreateUserRequest" in code
        assert "json.NewDecoder" in code
        assert "return &User" in code
    
    def test_generate_model_code(self, generator: CodeGenerator, sample_ir: Dict[str, Any]):
        """Test model code generation."""
        model = sample_ir["models"][0]
        code = generator._generate_model_code(model)
        
        # Check package and imports
        assert "package models" in code
        
        # Check struct definition
        assert "type User struct {" in code
        assert "ID" in code and 'json:"id"' in code
        assert "Name" in code and 'json:"name"' in code
        assert "Email" in code and 'json:"email"' in code
        
        # Check field types
        assert "int64" in code
        assert "string" in code
    
    def test_generate_router_code(self, generator: CodeGenerator, sample_ir: Dict[str, Any]):
        """Test router code generation."""
        routes = sample_ir["routes"]
        code = generator._generate_router_code(routes)
        
        # Check package and imports
        assert "package api" in code
        assert "github.com/gofiber/fiber/v2" in code
        
        # Check route setup
        assert "func SetupRoutes(app *fiber.App)" in code
        assert '"/users"' in code
        assert "app.Post" in code
        assert "CreateUserHandler" in code
    
    def test_generate_empty_ir(self, generator: CodeGenerator):
        """Test generating code from empty IR."""
        empty_ir = {
            "handlers": [],
            "models": [],
            "routes": []
        }
        
        context = {"ir": empty_ir}
        code = generator._generate_code(empty_ir)
        
        # Should still generate basic structure
        assert "handlers/base.go" in code
        assert "models/base.go" in code
        assert "api/router.go" in code
        
        # Check base files content
        assert "package handlers" in code["handlers/base.go"]
        assert "type Handler interface" in code["handlers/base.go"]
        
        assert "package models" in code["models/base.go"]
        assert "type Model interface" in code["models/base.go"]
        
        assert "package api" in code["api/router.go"]
        assert "func SetupRoutes" in code["api/router.go"]

class TestCodeOptimizer:
    """Test suite for CodeOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create an optimizer instance."""
        return CodeOptimizer()
    
    @pytest.fixture
    def sample_code(self) -> Dict[str, str]:
        """Sample generated code."""
        return {
            "handlers/user_handler.go": """
package handlers

import (
    "encoding/json"
    "net/http"
)

type CreateUserHandler struct {}

func (h *CreateUserHandler) Handle(w http.ResponseWriter, r *http.Request) {
    var request CreateUserRequest
    err := json.NewDecoder(r.Body).Decode(&request)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    // More code here...
}
""",
            "models/user.go": """
package models

type User struct {
    ID    int64  `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}
"""
        }
    
    @pytest.mark.asyncio
    async def test_optimize_code(
        self,
        optimizer: CodeOptimizer,
        sample_code: Dict[str, str]
    ):
        """Test optimizing code."""
        context = {"generated_code": sample_code}
        await optimizer.process(context)
        
        assert "optimized_code" in context
        optimized = context["optimized_code"]
        
        # Check handler optimization
        assert "handlers/user_handler.go" in optimized
        handler_code = optimized["handlers/user_handler.go"]
        assert "package handlers" in handler_code
        assert "import (" in handler_code  # Imports should be grouped
        assert "type CreateUserHandler struct" in handler_code
        
        # Check model optimization
        assert "models/user.go" in optimized
        model_code = optimized["models/user.go"]
        assert "package models" in model_code
        assert "type User struct" in model_code
    
    @pytest.mark.asyncio
    async def test_optimize_missing_code(self, optimizer: CodeOptimizer):
        """Test optimizing with missing code."""
        with pytest.raises(CompilationError) as exc_info:
            await optimizer.process({})
        
        assert exc_info.value.stage == CompilationStage.OPTIMIZATION
        assert "No generated code available" in str(exc_info.value)
    
    def test_optimize_file(self, optimizer: CodeOptimizer):
        """Test single file optimization."""
        code = """
package test

import "fmt"
import "strings"

func test() {
    x := "test"
    fmt.Println(x)
}
"""
        optimized = optimizer._optimize_file(code)
        
        # Check basic structure
        assert "package test" in optimized
        assert "func test()" in optimized
        
        # Check import grouping (ignoring exact whitespace)
        imports_section = [
            line.strip() for line in optimized.split("\n")
            if line.strip() and ("import" in line or "fmt" in line or "strings" in line)
        ]
        assert imports_section[0] == "import ("
        assert '"fmt"' in imports_section[1]
        assert '"strings"' in imports_section[2]
        assert ")" in imports_section[3]
        
        # Check code formatting
        assert 'x := "test"' in optimized
        assert "fmt.Println(x)" in optimized
    
    def test_optimize_empty_file(self, optimizer: CodeOptimizer):
        """Test optimizing empty file."""
        optimized = optimizer._optimize_file("")
        assert optimized == ""
    
    def test_optimize_invalid_code(self, optimizer: CodeOptimizer):
        """Test optimizing invalid Go code."""
        invalid_code = """
package test

func invalid( {
    return
"""
        # Should not raise exception, return original code
        optimized = optimizer._optimize_file(invalid_code)
        assert "func invalid" in optimized

class TestCodeValidator:
    """Test suite for CodeValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return CodeValidator()
    
    @pytest.fixture
    def valid_code(self) -> Dict[str, str]:
        """Sample valid code."""
        return {
            "handlers/user_handler.go": """
package handlers

import (
    "encoding/json"
    "net/http"
)

type CreateUserHandler struct {}

func (h *CreateUserHandler) Handle(w http.ResponseWriter, r *http.Request) {
    var request CreateUserRequest
    err := json.NewDecoder(r.Body).Decode(&request)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    // More code here...
}
""",
            "models/user.go": """
package models

type User struct {
    ID    int64  `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}
"""
        }
    
    @pytest.fixture
    def invalid_code(self) -> Dict[str, str]:
        """Sample invalid code."""
        return {
            "handlers/invalid_handler.go": """
package handlers

import (
    "encoding/json
    "net/http"
)

type InvalidHandler struct {

func (h *InvalidHandler) Handle(w http.ResponseWriter, r *http.Request) {
    var request Request
    err := json.NewDecoder(r.Body).Decode(&request
    if err != nil {
        return
    }
}
""",
            "models/invalid_model.go": """
package models

type InvalidModel struct {
    ID    int64  `json:"id`
    Name  string `json:"name
    Email string json:"email"`
}
"""
        }
    
    @pytest.mark.asyncio
    async def test_validate_code(
        self,
        validator: CodeValidator,
        valid_code: Dict[str, str]
    ):
        """Test validating code."""
        context = {"optimized_code": valid_code}
        await validator.process(context)
        
        assert "validation_results" in context
        results = context["validation_results"]
        assert results["valid"]
        assert not results["errors"]
        assert isinstance(results["warnings"], list)
    
    @pytest.mark.asyncio
    async def test_validate_missing_code(self, validator: CodeValidator):
        """Test validating with missing code."""
        with pytest.raises(CompilationError) as exc_info:
            await validator.process({})
        
        assert exc_info.value.stage == CompilationStage.VALIDATION
        assert "No optimized code available" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_invalid_code(
        self,
        validator: CodeValidator,
        invalid_code: Dict[str, str]
    ):
        """Test validating invalid code."""
        context = {"optimized_code": invalid_code}
        
        with pytest.raises(CompilationError) as exc_info:
            await validator.process(context)
        
        assert exc_info.value.stage == CompilationStage.VALIDATION
        assert "Validation failed" in str(exc_info.value)
        
        results = context["validation_results"]
        assert not results["valid"]
        assert len(results["errors"]) > 0
    
    def test_validate_file(self, validator: CodeValidator):
        """Test single file validation."""
        code = """
package test

import (
    "fmt"
    "strings"
)

func test() {
    x := "test"
    fmt.Println(x)
}
"""
        results = validator._validate_file("test.go", code)
        assert results["valid"]
        assert not results["errors"]
    
    def test_validate_empty_file(self, validator: CodeValidator):
        """Test validating empty file."""
        results = validator._validate_file("empty.go", "")
        assert not results["valid"]
        assert len(results["errors"]) > 0
        assert "empty file" in results["errors"][0].lower()
    
    def test_validate_syntax_errors(self, validator: CodeValidator):
        """Test validating code with syntax errors."""
        code = """
package test

func invalid( {
    return
"""
        results = validator._validate_file("invalid.go", code)
        assert not results["valid"]
        assert len(results["errors"]) > 0
        assert any("syntax error" in err.lower() for err in results["errors"])
    
    def test_validate_import_errors(self, validator: CodeValidator):
        """Test validating code with import errors."""
        code = """
package test

import (
    "nonexistent"
    "another/nonexistent"
)

func test() {}
"""
        results = validator._validate_file("imports.go", code)
        assert not results["valid"]
        assert len(results["errors"]) > 0
        assert any("import" in err.lower() for err in results["errors"]) 
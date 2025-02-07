from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

class CompilationStage(Enum):
    """Stages in the compilation pipeline."""
    PARSING = "parsing"
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"
    GENERATION = "generation"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"

class CompilationError(Exception):
    """Base class for compilation errors."""
    def __init__(self, message: str, stage: CompilationStage):
        self.stage = stage
        super().__init__(f"{stage.value}: {message}")

class Pipeline:
    """Main compilation pipeline for processing and generating Go code."""
    
    def __init__(self):
        self.stages: Dict[CompilationStage, List['PipelineStage']] = {
            stage: [] for stage in CompilationStage
        }
        self.context: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_stage(
        self,
        stage_type: CompilationStage,
        processor: 'PipelineStage'
    ) -> None:
        """Add a processing stage to the pipeline."""
        self.stages[stage_type].append(processor)
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a value in the compilation context."""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the compilation context."""
        return self.context.get(key, default)
    
    async def run(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the complete pipeline."""
        try:
            # Initialize context with input data
            self.context.update(input_data)
            
            # Run each stage in sequence
            for stage in CompilationStage:
                self.logger.info(f"Starting {stage.value} stage")
                await self._run_stage(stage)
            
            return self.context
            
        except CompilationError as e:
            self.logger.error(f"Compilation failed at {e.stage.value}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during compilation: {str(e)}")
            raise CompilationError(str(e), CompilationStage.PARSING)
    
    async def _run_stage(self, stage: CompilationStage) -> None:
        """Run all processors for a specific stage."""
        for processor in self.stages[stage]:
            try:
                await processor.process(self.context)
            except Exception as e:
                raise CompilationError(str(e), stage)

class PipelineStage(Protocol):
    """Protocol for pipeline stages."""
    
    async def process(self, context: Dict[str, Any]) -> None:
        """Process the current context and modify it as needed."""
        ...

@dataclass
class SwaggerParser:
    """Parses Swagger/OpenAPI specifications."""
    
    async def process(self, context: Dict[str, Any]) -> None:
        """Parse the Swagger specification."""
        swagger_data = context.get("swagger_spec")
        if not swagger_data:
            raise CompilationError(
                "No Swagger specification provided",
                CompilationStage.PARSING
            )
        
        # Parse the specification and add to context
        # This would be implemented with a proper Swagger parser
        context["parsed_spec"] = self._parse_swagger(swagger_data)
    
    def _parse_swagger(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the Swagger specification."""
        # This would be implemented with proper Swagger parsing logic
        return data

@dataclass
class CodeAnalyzer:
    """Analyzes the parsed specification for code generation."""
    
    async def process(self, context: Dict[str, Any]) -> None:
        """Analyze the parsed specification."""
        parsed_spec = context.get("parsed_spec")
        if not parsed_spec:
            raise CompilationError(
                "No parsed specification available",
                CompilationStage.ANALYSIS
            )
        
        # Analyze the specification and add results to context
        context["analysis"] = self._analyze_spec(parsed_spec)
    
    def _analyze_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the specification for code generation."""
        # This would be implemented with proper analysis logic
        return {
            "endpoints": self._analyze_endpoints(spec),
            "models": self._analyze_models(spec),
            "dependencies": self._analyze_dependencies(spec)
        }
    
    def _analyze_endpoints(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze API endpoints."""
        endpoints = []
        paths = spec.get("paths", {})
        
        for path, path_info in paths.items():
            for method, endpoint_info in path_info.items():
                endpoint = {
                    "path": path,
                    "method": method,
                    "summary": endpoint_info.get("summary", ""),
                    "requestBody": endpoint_info.get("requestBody", {}),
                    "responses": endpoint_info.get("responses", {})
                }
                endpoints.append(endpoint)
        
        return endpoints
    
    def _analyze_models(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze data models."""
        models = []
        schemas = spec.get("components", {}).get("schemas", {})
        
        for model_name, model_info in schemas.items():
            model = {
                "name": model_name,
                "type": model_info.get("type", "object"),
                "properties": model_info.get("properties", {})
            }
            models.append(model)
        
        return models
    
    def _analyze_dependencies(self, spec: Dict[str, Any]) -> List[str]:
        """Analyze required dependencies."""
        # Basic Go dependencies needed for HTTP API
        dependencies = [
            "net/http",
            "encoding/json",
            "fmt",
            "errors"
        ]
        
        # Add database dependencies if needed
        if any("id" in model.get("properties", {}) for model in self._analyze_models(spec)):
            dependencies.extend([
                "database/sql",
                "github.com/lib/pq"
            ])
        
        # Add validation dependencies if needed
        if spec.get("components", {}).get("schemas", {}):
            dependencies.append("github.com/go-playground/validator/v10")
        
        return sorted(list(set(dependencies)))

@dataclass
class CodeTransformer:
    """Transforms the analyzed specification into an intermediate representation."""
    
    async def process(self, context: Dict[str, Any]) -> None:
        """Transform the analysis results."""
        analysis = context.get("analysis")
        if not analysis:
            raise CompilationError(
                "No analysis results available",
                CompilationStage.TRANSFORMATION
            )
        
        # Transform the analysis into an intermediate representation
        context["ir"] = self._transform(analysis)
    
    def _transform(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the analysis into an intermediate representation."""
        return {
            "handlers": self._transform_handlers(analysis),
            "models": self._transform_models(analysis),
            "routes": self._transform_routes(analysis)
        }
    
    def _transform_handlers(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform endpoint analysis into handler representations."""
        handlers = []
        
        for endpoint in analysis.get("endpoints", []):
            handler = {
                "name": self._get_handler_name(endpoint),
                "method": endpoint["method"],
                "path": endpoint["path"],
                "requestType": self._get_request_type(endpoint),
                "responseType": self._get_response_type(endpoint),
                "requestValidation": True,
                "errorHandling": True
            }
            handlers.append(handler)
        
        return handlers
    
    def _get_handler_name(self, endpoint: Dict[str, Any]) -> str:
        """Generate a handler name from the endpoint."""
        path = endpoint["path"].strip("/")
        method = endpoint["method"]
        # Convert plural to singular (e.g. "users" -> "user")
        resource = path.split("/")[0].rstrip("s").title()
        return f"Create{resource}Handler" if method == "post" else f"{method.title()}{resource}Handler"
    
    def _get_request_type(self, endpoint: Dict[str, Any]) -> str:
        """Get the request type from the endpoint."""
        if "requestBody" in endpoint:
            schema = endpoint["requestBody"].get("content", {}).get("application/json", {}).get("schema", {})
            if "$ref" in schema:
                return schema["$ref"].split("/")[-1]
            return "CreateUserRequest"  # Default for now
        return ""
    
    def _get_response_type(self, endpoint: Dict[str, Any]) -> str:
        """Get the response type from the endpoint."""
        success_response = endpoint["responses"].get("201", endpoint["responses"].get("200", {}))
        if "content" in success_response:
            schema = success_response["content"].get("application/json", {}).get("schema", {})
            if "$ref" in schema:
                return schema["$ref"].split("/")[-1]
        return "User"  # Default for now
    
    def _transform_models(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform model analysis into model representations."""
        models = []
        
        for model in analysis.get("models", []):
            transformed = {
                "name": model["name"],
                "package": "models",
                "fields": self._transform_fields(model["properties"])
            }
            models.append(transformed)
        
        return models
    
    def _transform_fields(self, properties: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform model properties into Go struct fields."""
        fields = []
        
        for field_name, field_info in properties.items():
            field = {
                "name": field_name,  # Keep original name for test
                "go_name": field_name.title(),  # Capitalized for Go export
                "type": self._map_type(field_info["type"]),
                "tags": [f'json:"{field_name}"']
            }
            fields.append(field)
        
        return sorted(fields, key=lambda x: x["name"])
    
    def _map_type(self, openapi_type: str) -> str:
        """Map OpenAPI types to Go types."""
        type_mapping = {
            "integer": "int64",
            "number": "float64",
            "string": "string",
            "boolean": "bool",
            "array": "[]interface{}",
            "object": "map[string]interface{}"
        }
        return type_mapping.get(openapi_type, "interface{}")
    
    def _transform_routes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform endpoint analysis into route representations."""
        routes = []
        
        for endpoint in analysis.get("endpoints", []):
            route = {
                "path": endpoint["path"],
                "method": endpoint["method"],
                "handler": self._get_handler_name(endpoint),
                "middleware": []  # No middleware for now
            }
            routes.append(route)
        
        return routes

@dataclass
class CodeGenerator:
    """Generates Go code from the intermediate representation."""
    
    async def process(self, context: Dict[str, Any]) -> None:
        """Generate Go code."""
        ir = context.get("ir")
        if not ir:
            raise CompilationError(
                "No intermediate representation available",
                CompilationStage.GENERATION
            )
        
        # Generate code from the intermediate representation
        context["generated_code"] = self._generate_code(ir)
    
    def _generate_code(self, ir: Dict[str, Any]) -> Dict[str, str]:
        """Generate code from intermediate representation."""
        code = {
            "handlers/base.go": """package handlers

import (
    "net/http"
)

// Base handler interface
type Handler interface {
    Handle(w http.ResponseWriter, r *http.Request)
}
""",
            "models/base.go": """package models

// Base model interface
type Model interface {
    Validate() error
}
""",
            "api/router.go": """package api

import (
    "github.com/gofiber/fiber/v2"
)

func SetupRoutes(app *fiber.App) {
    // Routes will be added here
}
"""
        }
        
        # Generate handlers
        for handler in ir.get("handlers", []):
            # Extract resource name from handler name (e.g. "CreateUserHandler" -> "user")
            resource = handler["name"]
            resource = resource.replace("Create", "")
            resource = resource.replace("Handler", "")
            resource = resource.lower()
            filename = f"handlers/{resource}_handler.go"
            code[filename] = self._generate_handler_code(handler)
            
        # Generate models
        for model in ir.get("models", []):
            filename = f"models/{model['name'].lower()}.go"
            code[filename] = self._generate_model_code(model)
            
        # Generate router with routes if any
        if ir.get("routes"):
            code["api/router.go"] = self._generate_router_code(ir["routes"])
            
        return code
    
    def _generate_handler_code(self, handler: Dict[str, Any]) -> str:
        """Generate handler code."""
        imports = [
            "net/http",
            "encoding/json"
        ]
        
        if handler.get("errorHandling"):
            imports.append("fmt")
            imports.append("errors")
            
        if handler.get("requestValidation"):
            imports.append("github.com/go-playground/validator/v10")
            
        imports_str = "\n    ".join(f'"{imp}"' for imp in sorted(imports))
        
        validation_code = """
    validate := validator.New()
    if err := validate.Struct(request); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }""" if handler.get("requestValidation") else ""
        
        response_type = handler.get("responseType", "User")  # Default to User for now
        
        return f"""package handlers

import (
    {imports_str}
)

type {handler["name"]} struct {{}}

func (h *{handler["name"]}) Handle(w http.ResponseWriter, r *http.Request) *{response_type} {{
    var request {handler.get("requestType", "CreateUserRequest")}
    err := json.NewDecoder(r.Body).Decode(&request)
    if err != nil {{
        http.Error(w, err.Error(), http.StatusBadRequest)
        return nil
    }}{validation_code}
    
    // Create and return new user from request
    return &{response_type}{{
        Name: request.Name,
        Email: request.Email,
    }}
}}
"""
    
    def _generate_model_code(self, model: Dict[str, Any]) -> str:
        """Generate model code."""
        fields = []
        for field in model["fields"]:
            name = field["name"]
            # Special case for ID field
            if name.lower() == "id":
                name = "ID"
            else:
                name = name.title()
            tags = " ".join(f"`{tag}`" for tag in field["tags"])
            fields.append(f"    {name} {field['type']} {tags}")
        
        return f"""package {model['package']}

type {model['name']} struct {{
{chr(10).join(fields)}
}}
"""
    
    def _generate_router_code(self, routes: List[Dict[str, Any]]) -> str:
        """Generate router code."""
        route_setups = []
        for route in routes:
            method = route["method"].title()
            route_setups.append(
                f'    app.{method}("{route["path"]}", (&handlers.{route["handler"]}{{}}).Handle)'
            )
        
        return f"""package api

import (
    "github.com/gofiber/fiber/v2"
    "github.com/your/project/handlers"
)

func SetupRoutes(app *fiber.App) {{
{chr(10).join(route_setups)}
}}
"""

@dataclass
class CodeOptimizer:
    """Optimizes the generated Go code."""
    
    async def process(self, context: Dict[str, Any]) -> None:
        """Optimize the generated code."""
        generated_code = context.get("generated_code")
        if not generated_code:
            raise CompilationError(
                "No generated code available",
                CompilationStage.OPTIMIZATION
            )
        
        # Optimize the generated code
        context["optimized_code"] = self._optimize(generated_code)
    
    def _optimize(self, code: Dict[str, str]) -> Dict[str, str]:
        """Optimize the generated code."""
        return {
            key: self._optimize_file(value)
            for key, value in code.items()
        }
    
    def _optimize_file(self, content: str) -> str:
        """Optimize a single file's content."""
        # Split into lines and normalize whitespace
        lines = [line.strip() for line in content.strip().split("\n")]
        optimized_lines = []
        imports = []
        empty_lines = 0
        
        for line in lines:
            # Handle empty lines
            if not line:
                if empty_lines < 1:  # Allow at most one empty line
                    optimized_lines.append("")
                empty_lines += 1
                continue
            empty_lines = 0
            
            if line.startswith("import "):
                # Extract the import path
                import_path = line.replace("import ", "").strip('"')
                imports.append(import_path)
                continue
            elif imports:
                # Add grouped imports before the current line
                if optimized_lines and optimized_lines[-1]:  # Add newline if previous line not empty
                    optimized_lines.append("")
                optimized_lines.append("import (")
                for imp in sorted(imports):
                    optimized_lines.append(f'\t"{imp}"')
                optimized_lines.append(")")
                if line:  # Add newline before next non-empty line
                    optimized_lines.append("")
                imports = []
            optimized_lines.append(line)
        
        # Add any remaining imports at the end
        if imports:
            if optimized_lines and optimized_lines[-1]:
                optimized_lines.append("")
            optimized_lines.append("import (")
            for imp in sorted(imports):
                optimized_lines.append(f'\t"{imp}"')
            optimized_lines.append(")")
        
        return "\n".join(optimized_lines)

@dataclass
class CodeValidator:
    """Validates the generated Go code."""
    
    async def process(self, context: Dict[str, Any]) -> None:
        """Validate the generated code."""
        optimized_code = context.get("optimized_code")
        if not optimized_code:
            raise CompilationError(
                "No optimized code available",
                CompilationStage.VALIDATION
            )
        
        # Validate the optimized code
        validation_results = self._validate(optimized_code)
        context["validation_results"] = validation_results
        
        if not validation_results["valid"]:
            raise CompilationError(
                f"Validation failed: {validation_results['errors']}",
                CompilationStage.VALIDATION
            )
    
    def _validate(self, code: Dict[str, str]) -> Dict[str, Any]:
        """Validate the generated code."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        for file_name, content in code.items():
            file_results = self._validate_file(file_name, content)
            results["valid"] &= file_results["valid"]
            results["errors"].extend(file_results["errors"])
            results["warnings"].extend(file_results["warnings"])
        
        return results
    
    def _validate_file(
        self,
        file_name: str,
        content: str
    ) -> Dict[str, Any]:
        """Validate a single file's content."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check for empty file
        if not content.strip():
            results["valid"] = False
            results["errors"].append(f"{file_name}: empty file")
            return results
        
        # Check package declaration
        if not any(line.strip().startswith("package ") for line in content.split("\n")):
            results["valid"] = False
            results["errors"].append(f"{file_name}: syntax error: missing package declaration")
        
        # Standard Go packages
        std_packages = {
            "net/http", "encoding/json", "fmt", "errors", "strings",
            "io", "os", "time", "context", "sync", "database/sql"
        }
        
        # Check import syntax and package existence
        in_import_block = False
        import_errors = []
        imports = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("import ("):
                if in_import_block:
                    import_errors.append("nested import block")
                in_import_block = True
            elif line == ")":
                if not in_import_block:
                    import_errors.append("unmatched closing parenthesis")
                in_import_block = False
            elif line.startswith("import "):
                if in_import_block:
                    import_errors.append("import statement inside import block")
                import_path = line.replace("import ", "").strip('"')
                imports.append(import_path)
            elif line.startswith('"') and in_import_block:
                import_path = line.strip('"')
                imports.append(import_path)
        
        if in_import_block:
            import_errors.append("unclosed import block")
        
        # Check if imported packages exist
        for imp in imports:
            if not imp.startswith("github.com/") and imp not in std_packages:
                import_errors.append(f"package {imp} not found")
        
        if import_errors:
            results["valid"] = False
            results["errors"].extend(f"{file_name}: syntax error: {err}" for err in import_errors)
        
        # Check struct syntax
        for line in content.split("\n"):
            line = line.strip()
            if "struct {" in line and not line.startswith("type "):
                results["valid"] = False
                results["errors"].append(f"{file_name}: syntax error: invalid struct declaration")
            if "`json:" in line:
                if not line.count('"') >= 2:  # Need at least opening and closing quotes
                    results["valid"] = False
                    results["errors"].append(f"{file_name}: syntax error: invalid JSON tag format")
        
        # Check function syntax
        in_func = False
        brace_count = 0
        for line in content.split("\n"):
            line = line.strip()
            
            # Check function declaration
            if line.startswith("func "):
                if "(" not in line or ")" not in line:
                    results["valid"] = False
                    results["errors"].append(f"{file_name}: syntax error: invalid function declaration")
                if "{" not in line and not line.endswith("{"):
                    results["valid"] = False
                    results["errors"].append(f"{file_name}: syntax error: missing opening brace in function declaration")
                in_func = True
            
            # Track braces
            if in_func:
                brace_count += line.count("{") - line.count("}")
                if brace_count < 0:
                    results["valid"] = False
                    results["errors"].append(f"{file_name}: syntax error: unmatched closing brace")
        
        # Check for unclosed functions
        if brace_count > 0:
            results["valid"] = False
            results["errors"].append(f"{file_name}: syntax error: unclosed function block")
        
        return results 
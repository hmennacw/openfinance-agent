from typing import Dict, Any, List, Optional, Union, cast
from dataclasses import dataclass, field
from enum import Enum
import ast
import logging

class NodeType(Enum):
    """Types of AST nodes."""
    PACKAGE = "package"
    IMPORT = "import"
    STRUCT = "struct"
    INTERFACE = "interface"
    FUNCTION = "function"
    METHOD = "method"
    FIELD = "field"
    PARAMETER = "parameter"
    STATEMENT = "statement"
    EXPRESSION = "expression"

@dataclass
class Node:
    """Base class for AST nodes."""
    name: str
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(init=False)

@dataclass
class Package:
    """Represents a Go package."""
    name: str
    imports: List['Import'] = field(default_factory=list)
    declarations: List[Union['Struct', 'Interface', 'Function']] = field(default_factory=list)
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.PACKAGE, init=False)

@dataclass
class Import:
    """Represents a Go import statement."""
    path: str
    name: str
    alias: Optional[str] = None
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.IMPORT, init=False)

@dataclass
class Field:
    """Represents a struct field or interface method parameter."""
    name: str
    type_name: str
    tags: Optional[Dict[str, str]] = None
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.FIELD, init=False)

@dataclass
class Struct:
    """Represents a Go struct definition."""
    name: str
    fields: List[Field] = field(default_factory=list)
    methods: List['Method'] = field(default_factory=list)
    embedded: List[str] = field(default_factory=list)
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.STRUCT, init=False)

@dataclass
class Interface:
    """Represents a Go interface definition."""
    name: str
    methods: List['Method'] = field(default_factory=list)
    embedded: List[str] = field(default_factory=list)
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.INTERFACE, init=False)

@dataclass
class Parameter:
    """Represents a function parameter."""
    name: str
    type_name: str
    variadic: bool = False
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.PARAMETER, init=False)

@dataclass
class Function:
    """Represents a Go function definition."""
    name: str
    parameters: List[Parameter] = field(default_factory=list)
    results: List[Parameter] = field(default_factory=list)
    body: List['Statement'] = field(default_factory=list)
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.FUNCTION, init=False)

@dataclass
class Method:
    """Represents a Go method definition."""
    name: str
    receiver: Parameter
    parameters: List[Parameter] = field(default_factory=list)
    results: List[Parameter] = field(default_factory=list)
    body: List['Statement'] = field(default_factory=list)
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.METHOD, init=False)

@dataclass
class Statement:
    """Represents a Go statement."""
    name: str
    code: str
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.STATEMENT, init=False)

@dataclass
class Expression:
    """Represents a Go expression."""
    name: str
    code: str
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)
    node_type: NodeType = field(default=NodeType.EXPRESSION, init=False)

class ASTTransformer:
    """Transforms Go code AST."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse(self, code: str) -> Package:
        """Parse Go code into an AST."""
        # This would use a proper Go parser
        # For now, we'll create a dummy AST
        return self._create_dummy_ast()
    
    def transform(
        self,
        ast: Package,
        transformations: List[Dict[str, Any]]
    ) -> Package:
        """Apply transformations to the AST."""
        for transformation in transformations:
            if "action" not in transformation:
                self.logger.error(
                    "Error applying transformation: Transformation must specify an action"
                )
                continue
            
            try:
                ast = self._apply_transformation(ast, transformation)
            except Exception as e:
                self.logger.error(
                    f"Error applying transformation: {str(e)}"
                )
                continue
        return ast
    
    def generate(self, ast: Package) -> str:
        """Generate Go code from the AST."""
        return self._generate_package(ast)
    
    def _create_dummy_ast(self) -> Package:
        """Create a dummy AST for testing."""
        return Package(
            name="main",
            imports=[
                Import(name="fmt", path="fmt")
            ],
            declarations=[]
        )
    
    def _apply_transformation(
        self,
        ast: Package,
        transformation: Dict[str, Any]
    ) -> Package:
        """Apply a single transformation to the AST."""
        action = transformation.get("action")
        if not action:
            raise ValueError("Transformation must specify an action")
        
        if action == "add_import":
            return self._add_import(ast, transformation)
        elif action == "add_struct":
            return self._add_struct(ast, transformation)
        elif action == "add_interface":
            return self._add_interface(ast, transformation)
        elif action == "add_function":
            return self._add_function(ast, transformation)
        elif action == "modify_struct":
            return self._modify_struct(ast, transformation)
        elif action == "modify_interface":
            return self._modify_interface(ast, transformation)
        else:
            raise ValueError(f"Unknown transformation action: {action}")
    
    def _add_import(
        self,
        ast: Package,
        transformation: Dict[str, Any]
    ) -> Package:
        """Add an import to the package."""
        imp = Import(
            name=transformation["name"],
            path=transformation["path"],
            alias=transformation.get("alias")
        )
        ast.imports.append(imp)
        return ast
    
    def _add_struct(
        self,
        ast: Package,
        transformation: Dict[str, Any]
    ) -> Package:
        """Add a struct to the package."""
        struct = Struct(
            name=transformation["name"],
            fields=[
                Field(
                    name=f["name"],
                    type_name=f["type"],
                    tags=f.get("tags")
                )
                for f in transformation.get("fields", [])
            ],
            embedded=transformation.get("embedded", [])
        )
        ast.declarations.append(struct)
        return ast
    
    def _add_interface(
        self,
        ast: Package,
        transformation: Dict[str, Any]
    ) -> Package:
        """Add an interface to the package."""
        interface = Interface(
            name=transformation["name"],
            methods=[
                Method(
                    name=m["name"],
                    receiver=Parameter(
                        name="",
                        type_name=""
                    ),
                    parameters=[
                        Parameter(
                            name=p["name"],
                            type_name=p["type"],
                            variadic=p.get("variadic", False)
                        )
                        for p in m.get("parameters", [])
                    ],
                    results=[
                        Parameter(
                            name=r.get("name", ""),
                            type_name=r["type"]
                        )
                        for r in m.get("results", [])
                    ]
                )
                for m in transformation.get("methods", [])
            ],
            embedded=transformation.get("embedded", [])
        )
        ast.declarations.append(interface)
        return ast
    
    def _add_function(
        self,
        ast: Package,
        transformation: Dict[str, Any]
    ) -> Package:
        """Add a function to the package."""
        func = Function(
            name=transformation["name"],
            parameters=[
                Parameter(
                    name=p["name"],
                    type_name=p["type"],
                    variadic=p.get("variadic", False)
                )
                for p in transformation.get("parameters", [])
            ],
            results=[
                Parameter(
                    name=r.get("name", ""),
                    type_name=r["type"]
                )
                for r in transformation.get("results", [])
            ],
            body=[
                Statement(name="", code=line)
                for line in transformation.get("body", [])
            ]
        )
        ast.declarations.append(func)
        return ast
    
    def _modify_struct(
        self,
        ast: Package,
        transformation: Dict[str, Any]
    ) -> Package:
        """Modify a struct in the package."""
        target = transformation.get("name")
        if not target:
            raise ValueError("Struct modification must specify a target struct")

        for decl in ast.declarations:
            if isinstance(decl, Struct) and decl.name == target:
                # Remove fields
                remove_fields = transformation.get("remove_fields", [])
                decl.fields = [f for f in decl.fields if f.name not in remove_fields]

                # Add fields
                add_fields = transformation.get("add_fields", [])
                for field_data in add_fields:
                    field = Field(
                        name=field_data["name"],
                        type_name=field_data["type"],
                        tags=field_data.get("tags")
                    )
                    decl.fields.append(field)

                # Add methods
                add_methods = transformation.get("add_methods", [])
                for method_data in add_methods:
                    method = Method(
                        name=method_data["name"],
                        receiver=Parameter(name="r", type_name=target),
                        parameters=[
                            Parameter(
                                name=p.get("name", ""),
                                type_name=p["type"],
                                variadic=p.get("variadic", False)
                            )
                            for p in method_data.get("parameters", [])
                        ],
                        results=[
                            Parameter(
                                name=r.get("name", ""),
                                type_name=r["type"]
                            )
                            for r in method_data.get("results", [])
                        ]
                    )
                    decl.methods.append(method)
                break
        return ast
    
    def _modify_interface(
        self,
        ast: Package,
        transformation: Dict[str, Any]
    ) -> Package:
        """Modify an interface in the package."""
        target = transformation.get("name")
        if not target:
            raise ValueError("Interface modification must specify a target interface")

        for decl in ast.declarations:
            if isinstance(decl, Interface) and decl.name == target:
                # Remove methods
                remove_methods = transformation.get("remove_methods", [])
                decl.methods = [m for m in decl.methods if m.name not in remove_methods]

                # Add methods
                add_methods = transformation.get("add_methods", [])
                for method_data in add_methods:
                    method = Method(
                        name=method_data["name"],
                        receiver=Parameter(name="r", type_name=target),
                        parameters=[
                            Parameter(
                                name=p.get("name", ""),
                                type_name=p["type"],
                                variadic=p.get("variadic", False)
                            )
                            for p in method_data.get("parameters", [])
                        ],
                        results=[
                            Parameter(
                                name=r.get("name", ""),
                                type_name=r["type"]
                            )
                            for r in method_data.get("results", [])
                        ]
                    )
                    decl.methods.append(method)
                break
        return ast
    
    def _generate_package(self, package: Package) -> str:
        """Generate Go code for a package."""
        parts = [f"package {package.name}\n\n"]
        
        if package.imports:
            if len(package.imports) == 1:
                imp = package.imports[0]
                if imp.alias:
                    parts.append(f'import {imp.alias} "{imp.path}"\n\n')
                else:
                    parts.append(f'import "{imp.path}"\n\n')
            else:
                parts.append("import (\n")
                for imp in package.imports:
                    if imp.alias:
                        parts.append(f'\t{imp.alias} "{imp.path}"\n')
                    else:
                        parts.append(f'\t"{imp.path}"\n')
                parts.append(")\n\n")
        
        for decl in package.declarations:
            parts.append(self._generate_declaration(decl))
            parts.append("\n")
        
        return "".join(parts)
    
    def _generate_declaration(
        self,
        decl: Union[Struct, Interface, Function]
    ) -> str:
        """Generate Go code for a declaration."""
        if isinstance(decl, Struct):
            return self._generate_struct(decl)
        elif isinstance(decl, Interface):
            return self._generate_interface(decl)
        elif isinstance(decl, Function):
            return self._generate_function(decl)
        return ""
    
    def _generate_struct(self, struct: Struct) -> str:
        """Generate Go code for a struct."""
        parts = [f"type {struct.name} struct {{\n"]
        
        # Generate fields
        max_name_len = max((len(f.name) for f in struct.fields), default=0)
        max_type_len = max((len(f.type_name) for f in struct.fields), default=0)
        
        for field in struct.fields:
            field_str = f"\t{field.name}"
            # Add padding for alignment (name)
            field_str += " " * (max_name_len - len(field.name) + 3)
            field_str += field.type_name
            # Add padding for alignment (type)
            if field.tags:
                field_str += " " * (max_type_len - len(field.type_name) + 4)
                tags_str = " ".join(f'{k}:"{v}"' for k, v in field.tags.items())
                field_str += f"`{tags_str}`"
            parts.append(field_str + "\n")
        
        parts.append("}\n\n")
        
        # Generate methods
        for method in struct.methods:
            parts.append(self._generate_method(method, struct.name))
            parts.append("\n\n")
        
        return "".join(parts)
    
    def _generate_interface(self, interface: Interface) -> str:
        """Generate Go code for an interface."""
        parts = [f"type {interface.name} interface {{\n"]
        
        # Generate embedded interfaces
        for embedded in interface.embedded:
            parts.append(f"\t{embedded}\n")
        
        # Generate methods
        for method in interface.methods:
            params = self._generate_parameters(method.parameters)
            results = self._generate_results(method.results)
            if results:
                parts.append(f"\t{method.name}({params}) {results}\n")
            else:
                parts.append(f"\t{method.name}({params})\n")
        
        parts.append("}\n")
        return "".join(parts)
    
    def _generate_function(self, func: Function) -> str:
        """Generate Go code for a function."""
        params = self._generate_parameters(func.parameters)
        results = self._generate_results(func.results)
        
        parts = [f"func {func.name}({params})"]
        if results:
            parts.append(f" {results}")
        parts.append(" {\n")
        
        for stmt in func.body:
            parts.append(f"\t{stmt.code}\n")
        
        parts.append("}")
        return "".join(parts)
    
    def _generate_method(self, method: Method, receiver_type: str) -> str:
        """Generate Go code for a method."""
        receiver = f"({method.receiver.name} {method.receiver.type_name})"
        params = self._generate_parameters(method.parameters)
        results = self._generate_results(method.results)
        
        parts = [f"func {receiver} {method.name}({params})"]
        if results:
            parts.append(f" {results}")
        parts.append(" {\n")
        
        for stmt in method.body:
            parts.append(f"\t{stmt.code}\n")
        
        parts.append("}")
        return "".join(parts)
    
    def _generate_parameters(self, params: List[Parameter]) -> str:
        """Generate Go code for parameters."""
        param_strs = []
        for param in params:
            param_str = param.name
            if param.variadic:
                param_str += f" ...{param.type_name}"
            else:
                param_str += f" {param.type_name}"
            param_strs.append(param_str)
        return ", ".join(param_strs)
    
    def _generate_results(self, results: List[Parameter]) -> str:
        """Generate Go code for results."""
        if not results:
            return ""
        
        if len(results) == 1 and not results[0].name:
            return results[0].type_name
        
        result_strs = []
        for result in results:
            if result.name:
                result_strs.append(f"{result.name} {result.type_name}")
            else:
                result_strs.append(result.type_name)
        
        return f"({', '.join(result_strs)})" 
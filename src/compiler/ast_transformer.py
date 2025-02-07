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
    node_type: NodeType
    name: str
    position: Optional[Dict[str, int]] = None
    comments: List[str] = field(default_factory=list)

@dataclass
class Package(Node):
    """Represents a Go package."""
    imports: List['Import'] = field(default_factory=list)
    declarations: List[Union['Struct', 'Interface', 'Function']] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = NodeType.PACKAGE

@dataclass
class Import(Node):
    """Represents a Go import statement."""
    path: str
    alias: Optional[str] = None
    
    def __post_init__(self):
        self.node_type = NodeType.IMPORT

@dataclass
class Field(Node):
    """Represents a struct field or interface method parameter."""
    type_name: str
    tags: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        self.node_type = NodeType.FIELD

@dataclass
class Struct(Node):
    """Represents a Go struct definition."""
    fields: List[Field] = field(default_factory=list)
    methods: List['Method'] = field(default_factory=list)
    embedded: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = NodeType.STRUCT

@dataclass
class Interface(Node):
    """Represents a Go interface definition."""
    methods: List['Method'] = field(default_factory=list)
    embedded: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = NodeType.INTERFACE

@dataclass
class Parameter(Node):
    """Represents a function parameter."""
    type_name: str
    variadic: bool = False
    
    def __post_init__(self):
        self.node_type = NodeType.PARAMETER

@dataclass
class Function(Node):
    """Represents a Go function definition."""
    parameters: List[Parameter] = field(default_factory=list)
    results: List[Parameter] = field(default_factory=list)
    body: List['Statement'] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = NodeType.FUNCTION

@dataclass
class Method(Function):
    """Represents a Go method definition."""
    receiver: Parameter
    
    def __post_init__(self):
        self.node_type = NodeType.METHOD

@dataclass
class Statement(Node):
    """Represents a Go statement."""
    code: str
    
    def __post_init__(self):
        self.node_type = NodeType.STATEMENT

@dataclass
class Expression(Node):
    """Represents a Go expression."""
    code: str
    
    def __post_init__(self):
        self.node_type = NodeType.EXPRESSION

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
            try:
                ast = self._apply_transformation(ast, transformation)
            except Exception as e:
                self.logger.error(
                    f"Error applying transformation: {str(e)}"
                )
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
        """Modify an existing struct in the package."""
        struct_name = transformation["target"]
        for i, decl in enumerate(ast.declarations):
            if isinstance(decl, Struct) and decl.name == struct_name:
                struct = cast(Struct, decl)
                
                # Add new fields
                for field in transformation.get("add_fields", []):
                    struct.fields.append(
                        Field(
                            name=field["name"],
                            type_name=field["type"],
                            tags=field.get("tags")
                        )
                    )
                
                # Remove fields
                for field_name in transformation.get("remove_fields", []):
                    struct.fields = [
                        f for f in struct.fields
                        if f.name != field_name
                    ]
                
                ast.declarations[i] = struct
                break
        
        return ast
    
    def _modify_interface(
        self,
        ast: Package,
        transformation: Dict[str, Any]
    ) -> Package:
        """Modify an existing interface in the package."""
        interface_name = transformation["target"]
        for i, decl in enumerate(ast.declarations):
            if isinstance(decl, Interface) and decl.name == interface_name:
                interface = cast(Interface, decl)
                
                # Add new methods
                for method in transformation.get("add_methods", []):
                    interface.methods.append(
                        Method(
                            name=method["name"],
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
                                for p in method.get("parameters", [])
                            ],
                            results=[
                                Parameter(
                                    name=r.get("name", ""),
                                    type_name=r["type"]
                                )
                                for r in method.get("results", [])
                            ]
                        )
                    )
                
                # Remove methods
                for method_name in transformation.get("remove_methods", []):
                    interface.methods = [
                        m for m in interface.methods
                        if m.name != method_name
                    ]
                
                ast.declarations[i] = interface
                break
        
        return ast
    
    def _generate_package(self, package: Package) -> str:
        """Generate Go code for a package."""
        lines = [
            f"package {package.name}\n",
            self._generate_imports(package.imports),
            "\n".join(
                self._generate_declaration(decl)
                for decl in package.declarations
            )
        ]
        return "\n\n".join(filter(None, lines))
    
    def _generate_imports(self, imports: List[Import]) -> str:
        """Generate Go code for imports."""
        if not imports:
            return ""
        
        if len(imports) == 1:
            imp = imports[0]
            return f'import {imp.alias + " " if imp.alias else ""}{imp.path}'
        
        lines = ["import ("]
        for imp in imports:
            lines.append(
                f'\t{imp.alias + " " if imp.alias else ""}{imp.path}'
            )
        lines.append(")")
        return "\n".join(lines)
    
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
        lines = [f"type {struct.name} struct {{"]
        
        # Add embedded types
        for embedded in struct.embedded:
            lines.append(f"\t{embedded}")
        
        # Add fields
        for field in struct.fields:
            tag_str = ""
            if field.tags:
                tags = " ".join(
                    f'{key}:"{value}"'
                    for key, value in field.tags.items()
                )
                tag_str = f" `{tags}`"
            lines.append(f"\t{field.name} {field.type_name}{tag_str}")
        
        lines.append("}")
        
        # Add methods
        if struct.methods:
            lines.append("")
            for method in struct.methods:
                lines.append(self._generate_method(method, struct.name))
        
        return "\n".join(lines)
    
    def _generate_interface(self, interface: Interface) -> str:
        """Generate Go code for an interface."""
        lines = [f"type {interface.name} interface {{"]
        
        # Add embedded interfaces
        for embedded in interface.embedded:
            lines.append(f"\t{embedded}")
        
        # Add methods
        for method in interface.methods:
            params = self._generate_parameters(method.parameters)
            results = self._generate_results(method.results)
            lines.append(f"\t{method.name}({params}) {results}")
        
        lines.append("}")
        return "\n".join(lines)
    
    def _generate_function(self, func: Function) -> str:
        """Generate Go code for a function."""
        params = self._generate_parameters(func.parameters)
        results = self._generate_results(func.results)
        
        lines = [f"func {func.name}({params}) {results} {{"]
        for stmt in func.body:
            lines.append(f"\t{stmt.code}")
        lines.append("}")
        return "\n".join(lines)
    
    def _generate_method(self, method: Method, receiver_type: str) -> str:
        """Generate Go code for a method."""
        params = self._generate_parameters(method.parameters)
        results = self._generate_results(method.results)
        
        lines = [
            f"func (r {receiver_type}) {method.name}({params}) {results} {{"
        ]
        for stmt in method.body:
            lines.append(f"\t{stmt.code}")
        lines.append("}")
        return "\n".join(lines)
    
    def _generate_parameters(self, params: List[Parameter]) -> str:
        """Generate Go code for function parameters."""
        return ", ".join(
            f"{p.name} {'...' if p.variadic else ''}{p.type_name}"
            for p in params
        )
    
    def _generate_results(self, results: List[Parameter]) -> str:
        """Generate Go code for function results."""
        if not results:
            return ""
        elif len(results) == 1 and not results[0].name:
            return results[0].type_name
        
        return "(" + ", ".join(
            f"{r.name} {r.type_name}" if r.name else r.type_name
            for r in results
        ) + ")" 
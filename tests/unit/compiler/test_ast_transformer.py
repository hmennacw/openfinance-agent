import pytest
from typing import Dict, Any, List
from src.compiler.ast_transformer import (
    NodeType,
    Node,
    Package,
    Import,
    Field,
    Struct,
    Interface,
    Parameter,
    Function,
    Method,
    Statement,
    Expression,
    ASTTransformer,
)

def test_node_types():
    """Test that all node types are correctly defined."""
    assert NodeType.PACKAGE.value == "package"
    assert NodeType.IMPORT.value == "import"
    assert NodeType.STRUCT.value == "struct"
    assert NodeType.INTERFACE.value == "interface"
    assert NodeType.FUNCTION.value == "function"
    assert NodeType.METHOD.value == "method"
    assert NodeType.FIELD.value == "field"
    assert NodeType.PARAMETER.value == "parameter"
    assert NodeType.STATEMENT.value == "statement"
    assert NodeType.EXPRESSION.value == "expression"

def test_base_node():
    """Test base Node class initialization and properties."""
    node = Node(name="test")
    assert node.name == "test"
    assert node.position is None
    assert node.comments == []

    # Test with optional parameters
    node = Node(
        name="test",
        position={"line": 1, "column": 0},
        comments=["Test comment"]
    )
    assert node.position == {"line": 1, "column": 0}
    assert node.comments == ["Test comment"]

def test_package():
    """Test Package class initialization and post_init."""
    package = Package(name="main")
    assert package.node_type == NodeType.PACKAGE
    assert package.name == "main"
    assert package.imports == []
    assert package.declarations == []

def test_import():
    """Test Import class initialization and post_init."""
    imp = Import(name="fmt", path="fmt")
    assert imp.node_type == NodeType.IMPORT
    assert imp.name == "fmt"
    assert imp.path == "fmt"
    assert imp.alias is None

    imp_with_alias = Import(name="fmt", path="fmt", alias="f")
    assert imp_with_alias.alias == "f"

def test_field():
    """Test Field class initialization and post_init."""
    field = Field(name="ID", type_name="int")
    assert field.node_type == NodeType.FIELD
    assert field.name == "ID"
    assert field.type_name == "int"
    assert field.tags is None

    field_with_tags = Field(
        name="ID",
        type_name="int",
        tags={"json": "id"}
    )
    assert field_with_tags.tags == {"json": "id"}

def test_struct():
    """Test Struct class initialization and post_init."""
    struct = Struct(name="User")
    assert struct.node_type == NodeType.STRUCT
    assert struct.name == "User"
    assert struct.fields == []
    assert struct.methods == []
    assert struct.embedded == []

def test_interface():
    """Test Interface class initialization and post_init."""
    interface = Interface(name="Repository")
    assert interface.node_type == NodeType.INTERFACE
    assert interface.name == "Repository"
    assert interface.methods == []
    assert interface.embedded == []

def test_parameter():
    """Test Parameter class initialization and post_init."""
    param = Parameter(name="id", type_name="int")
    assert param.node_type == NodeType.PARAMETER
    assert param.name == "id"
    assert param.type_name == "int"
    assert not param.variadic

    variadic_param = Parameter(name="args", type_name="string", variadic=True)
    assert variadic_param.variadic

def test_function():
    """Test Function class initialization and post_init."""
    func = Function(name="GetUser")
    assert func.node_type == NodeType.FUNCTION
    assert func.name == "GetUser"
    assert func.parameters == []
    assert func.results == []
    assert func.body == []

def test_method():
    """Test Method class initialization and post_init."""
    receiver = Parameter(name="r", type_name="Repository")
    method = Method(name="Find", receiver=receiver)
    assert method.node_type == NodeType.METHOD
    assert method.name == "Find"
    assert method.receiver == receiver
    assert method.parameters == []
    assert method.results == []
    assert method.body == []

def test_statement():
    """Test Statement class initialization and post_init."""
    stmt = Statement(name="return", code="return nil")
    assert stmt.node_type == NodeType.STATEMENT
    assert stmt.name == "return"
    assert stmt.code == "return nil"

def test_expression():
    """Test Expression class initialization and post_init."""
    expr = Expression(name="call", code="repo.Find()")
    assert expr.node_type == NodeType.EXPRESSION
    assert expr.name == "call"
    assert expr.code == "repo.Find()"

class TestASTTransformer:
    @pytest.fixture
    def transformer(self):
        return ASTTransformer()

    def test_init(self, transformer):
        """Test ASTTransformer initialization."""
        assert transformer.logger is not None

    def test_parse(self, transformer):
        """Test parsing code into AST."""
        ast = transformer.parse("package main")
        assert isinstance(ast, Package)
        assert ast.name == "main"
        assert len(ast.imports) == 1
        assert ast.imports[0].name == "fmt"

    def test_transform_add_import(self, transformer):
        """Test adding import transformation."""
        ast = Package(name="main")
        transformation = {
            "action": "add_import",
            "name": "time",
            "path": "time"
        }
        result = transformer.transform(ast, [transformation])
        assert len(result.imports) == 1
        assert result.imports[0].name == "time"
        assert result.imports[0].path == "time"

    def test_transform_add_struct(self, transformer):
        """Test adding struct transformation."""
        ast = Package(name="main")
        transformation = {
            "action": "add_struct",
            "name": "User",
            "fields": [
                {"name": "ID", "type": "int"},
                {"name": "Name", "type": "string", "tags": {"json": "name"}}
            ],
            "embedded": ["BaseModel"]
        }
        result = transformer.transform(ast, [transformation])
        assert len(result.declarations) == 1
        struct = result.declarations[0]
        assert isinstance(struct, Struct)
        assert struct.name == "User"
        assert len(struct.fields) == 2
        assert struct.embedded == ["BaseModel"]

    def test_transform_add_interface(self, transformer):
        """Test adding interface transformation."""
        ast = Package(name="main")
        transformation = {
            "action": "add_interface",
            "name": "Repository",
            "methods": [
                {
                    "name": "Find",
                    "parameters": [{"name": "id", "type": "int"}],
                    "results": [{"type": "error"}]
                }
            ]
        }
        result = transformer.transform(ast, [transformation])
        assert len(result.declarations) == 1
        interface = result.declarations[0]
        assert isinstance(interface, Interface)
        assert interface.name == "Repository"
        assert len(interface.methods) == 1

    def test_transform_invalid_action(self, transformer):
        """Test transform with invalid action."""
        ast = Package(name="main")
        transformation = {
            "action": "invalid_action"
        }
        result = transformer.transform(ast, [transformation])
        assert result == ast  # AST should be unchanged

    def test_transform_missing_action(self, transformer):
        """Test transform with missing action."""
        ast = Package(name="main")
        transformation = {}
        result = transformer.transform(ast, [transformation])
        assert result == ast  # AST should be unchanged

    def test_transform_error_handling(self, transformer):
        """Test transform error handling."""
        ast = Package(name="main")
        transformation = {
            "action": "add_import",
            # Missing required fields to trigger error
        }
        # Should log error but not raise exception
        result = transformer.transform(ast, [transformation])
        assert result == ast  # Original AST should be unchanged

    def test_transform_add_function(self, transformer):
        """Test adding function transformation."""
        ast = Package(name="main")
        transformation = {
            "action": "add_function",
            "name": "GetUser",
            "parameters": [
                {"name": "id", "type": "int"}
            ],
            "results": [
                {"name": "user", "type": "*User"},
                {"type": "error"}
            ],
            "body": [
                {"code": "return nil, nil"}
            ]
        }
        result = transformer.transform(ast, [transformation])
        assert len(result.declarations) == 1
        func = result.declarations[0]
        assert isinstance(func, Function)
        assert func.name == "GetUser"
        assert len(func.parameters) == 1
        assert len(func.results) == 2
        assert len(func.body) == 1

    def test_transform_modify_struct(self, transformer):
        """Test modifying struct transformation."""
        ast = Package(name="main")
        struct = Struct(
            name="User",
            fields=[Field(name="ID", type_name="int")]
        )
        ast.declarations.append(struct)

        transformation = {
            "action": "modify_struct",
            "name": "User",
            "add_fields": [
                {"name": "Name", "type": "string"}
            ],
            "remove_fields": ["ID"],
            "add_methods": [
                {
                    "name": "GetName",
                    "results": [{"type": "string"}]
                }
            ]
        }
        result = transformer.transform(ast, [transformation])
        modified_struct = result.declarations[0]
        assert isinstance(modified_struct, Struct)
        assert len(modified_struct.fields) == 1
        assert modified_struct.fields[0].name == "Name"
        assert len(modified_struct.methods) == 1
        assert modified_struct.methods[0].name == "GetName"

    def test_transform_modify_interface(self, transformer):
        """Test modifying interface transformation."""
        ast = Package(name="main")
        interface = Interface(
            name="Repository",
            methods=[
                Method(
                    name="Find",
                    receiver=Parameter(name="r", type_name="Repository"),
                    parameters=[Parameter(name="id", type_name="int")],
                    results=[Parameter(name="", type_name="error")]
                )
            ]
        )
        ast.declarations.append(interface)

        transformation = {
            "action": "modify_interface",
            "name": "Repository",
            "add_methods": [
                {
                    "name": "Save",
                    "parameters": [{"name": "entity", "type": "interface{}"}],
                    "results": [{"type": "error"}]
                }
            ],
            "remove_methods": ["Find"]
        }
        result = transformer.transform(ast, [transformation])
        modified_interface = result.declarations[0]
        assert isinstance(modified_interface, Interface)
        assert len(modified_interface.methods) == 1
        assert modified_interface.methods[0].name == "Save"

    def test_generate(self, transformer):
        """Test code generation from AST."""
        ast = Package(
            name="main",
            imports=[Import(name="fmt", path="fmt")],
            declarations=[
                Struct(
                    name="User",
                    fields=[Field(name="ID", type_name="int")]
                )
            ]
        )
        code = transformer.generate(ast)
        assert isinstance(code, str)
        assert "package main" in code
        assert "import" in code
        assert "type User struct" in code

    def test_generate_package(self, transformer):
        """Test package code generation."""
        ast = Package(
            name="main",
            imports=[
                Import(name="fmt", path="fmt"),
                Import(name="io", path="io", alias="stdio")
            ]
        )
        code = transformer.generate(ast)
        assert "package main" in code
        assert '"fmt"' in code
        assert 'stdio "io"' in code

    def test_generate_struct_with_methods(self, transformer):
        """Test struct with methods code generation."""
        ast = Package(name="main")
        struct = Struct(
            name="User",
            fields=[
                Field(name="ID", type_name="int", tags={"json": "id"}),
                Field(name="Name", type_name="string", tags={"json": "name"})
            ]
        )
        method = Method(
            name="GetName",
            receiver=Parameter(name="u", type_name="*User"),
            results=[Parameter(name="", type_name="string")]
        )
        struct.methods.append(method)
        ast.declarations.append(struct)

        code = transformer.generate(ast)
        assert "type User struct" in code
        assert "ID" in code and "int" in code and '`json:"id"`' in code
        assert "Name" in code and "string" in code and '`json:"name"`' in code
        assert "func (u *User) GetName() string" in code

    def test_generate_interface_with_embedded(self, transformer):
        """Test interface with embedded interfaces code generation."""
        ast = Package(name="main")
        interface = Interface(
            name="Repository",
            embedded=["io.Reader", "io.Writer"],
            methods=[
                Method(
                    name="Find",
                    receiver=Parameter(name="r", type_name="Repository"),
                    parameters=[Parameter(name="id", type_name="int")],
                    results=[
                        Parameter(name="result", type_name="interface{}"),
                        Parameter(name="", type_name="error")
                    ]
                )
            ]
        )
        ast.declarations.append(interface)

        code = transformer.generate(ast)
        assert "type Repository interface" in code
        assert "io.Reader" in code
        assert "io.Writer" in code
        assert "Find(id int) (result interface{}, error)" in code

    def test_generate_function_with_body(self, transformer):
        """Test function with body code generation."""
        ast = Package(name="main")
        function = Function(
            name="main",
            body=[
                Statement(name="print", code="fmt.Println(\"Hello, World!\")"),
                Statement(name="return", code="return")
            ]
        )
        ast.declarations.append(function)

        code = transformer.generate(ast)
        assert "func main()" in code
        assert "fmt.Println(\"Hello, World!\")" in code
        assert "return" in code

    def test_generate_parameters(self, transformer):
        """Test parameters code generation."""
        params = [
            Parameter(name="ctx", type_name="context.Context"),
            Parameter(name="id", type_name="int"),
            Parameter(name="args", type_name="string", variadic=True)
        ]
        code = transformer._generate_parameters(params)
        assert "ctx context.Context, id int, args ...string" in code

    def test_generate_results(self, transformer):
        """Test results code generation."""
        # Single unnamed result
        results = [Parameter(name="", type_name="error")]
        code = transformer._generate_results(results)
        assert code == "error"

        # Multiple named results
        results = [
            Parameter(name="result", type_name="*User"),
            Parameter(name="err", type_name="error")
        ]
        code = transformer._generate_results(results)
        assert code == "(result *User, err error)"

        # Multiple unnamed results
        results = [
            Parameter(name="", type_name="*User"),
            Parameter(name="", type_name="error")
        ]
        code = transformer._generate_results(results)
        assert code == "(*User, error)" 
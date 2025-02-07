import pytest
from pathlib import Path
from typing import Dict, Any

@pytest.fixture
def sample_swagger_spec() -> Dict[str, Any]:
    """Sample Swagger specification for testing."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0"
        },
        "paths": {
            "/users": {
                "get": {
                    "summary": "List users",
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/UserList"
                                    }
                                }
                            }
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
                },
                "UserList": {
                    "type": "array",
                    "items": {
                        "$ref": "#/components/schemas/User"
                    }
                }
            }
        }
    }

@pytest.fixture
def temp_storage_path(tmp_path: Path) -> Path:
    """Temporary storage path for testing."""
    return tmp_path / "test_storage"

@pytest.fixture
def sample_code_memory() -> Dict[str, Any]:
    """Sample code memory for testing."""
    return {
        "file_path": "handlers/user_handler.go",
        "code_type": "handler",
        "content": """
package handlers

import "github.com/gofiber/fiber/v2"

func GetUsers(c *fiber.Ctx) error {
    return c.JSON([]User{})
}
""",
        "dependencies": ["models.User"]
    }

@pytest.fixture
def sample_learning_example() -> Dict[str, Any]:
    """Sample learning example for testing."""
    return {
        "id": "test_example",
        "context": {
            "task_type": "generate_handler",
            "endpoint": "/users",
            "method": "GET"
        },
        "decision": {
            "chosen_option": "fiber_handler",
            "confidence": 0.85
        },
        "outcome": {
            "status": "completed",
            "validation_passed": True
        },
        "tags": ["handler", "success"]
    } 
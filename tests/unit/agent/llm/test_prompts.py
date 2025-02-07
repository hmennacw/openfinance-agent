import pytest
from typing import Dict, Any

from src.agent.llm.prompts import PromptTemplate, PromptLibrary, PromptManager

class TestPromptTemplate:
    """Test suite for PromptTemplate."""
    
    def test_format_with_all_variables(self):
        """Test formatting with all required variables."""
        template = PromptTemplate(
            template="Hello ${name}, welcome to ${place}!",
            required_variables={"name", "place"}
        )
        
        result = template.format(name="John", place="Earth")
        assert result == "Hello John, welcome to Earth!"
    
    def test_format_with_extra_variables(self):
        """Test formatting with extra variables."""
        template = PromptTemplate(
            template="Hello ${name}!",
            required_variables={"name"}
        )
        
        result = template.format(name="John", extra="ignored")
        assert result == "Hello John!"
    
    def test_format_with_missing_variables(self):
        """Test formatting with missing variables."""
        template = PromptTemplate(
            template="Hello ${name}, welcome to ${place}!",
            required_variables={"name", "place"}
        )
        
        with pytest.raises(ValueError) as exc_info:
            template.format(name="John")
        assert "Missing required variables" in str(exc_info.value)
        assert "place" in str(exc_info.value)

class TestPromptLibrary:
    """Test suite for PromptLibrary."""
    
    @pytest.fixture
    def library(self):
        """Create a PromptLibrary instance."""
        return PromptLibrary()
    
    def test_go_handler_template(self, library):
        """Test Go handler template."""
        template = library.go_handler_template
        assert isinstance(template, PromptTemplate)
        assert template.required_variables == {"path", "method", "description"}
        
        # Test formatting
        result = template.format(
            path="/users",
            method="POST",
            description="Create a new user",
            context={"extra": "info"}
        )
        assert "/users" in result
        assert "POST" in result
        assert "Create a new user" in result
    
    def test_go_usecase_template(self, library):
        """Test Go usecase template."""
        template = library.go_usecase_template
        assert isinstance(template, PromptTemplate)
        assert template.required_variables == {"name", "description"}
        
        # Test formatting
        result = template.format(
            name="CreateUser",
            description="Create a new user in the system",
            context={"extra": "info"}
        )
        assert "CreateUser" in result
        assert "Create a new user in the system" in result
    
    def test_go_model_template(self, library):
        """Test Go model template."""
        template = library.go_model_template
        assert isinstance(template, PromptTemplate)
        assert template.required_variables == {"schema"}
        
        # Test formatting
        result = template.format(
            schema="type User struct { ID int }",
            context={"extra": "info"}
        )
        assert "type User struct { ID int }" in result
    
    def test_code_review_template(self, library):
        """Test code review template."""
        template = library.code_review_template
        assert isinstance(template, PromptTemplate)
        assert template.required_variables == {"code"}
        
        # Test formatting
        result = template.format(
            code="func main() {}",
            context={"extra": "info"}
        )
        assert "func main() {}" in result

class TestPromptManager:
    """Test suite for PromptManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a PromptManager instance."""
        return PromptManager()
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert isinstance(manager.library, PromptLibrary)
    
    def test_generate_handler_prompt(self, manager):
        """Test handler prompt generation."""
        prompt = manager.generate_handler_prompt(
            path="/users",
            method="POST",
            description="Create a new user",
            context={"extra": "info"}
        )
        assert "/users" in prompt
        assert "POST" in prompt
        assert "Create a new user" in prompt
    
    def test_generate_handler_prompt_without_context(self, manager):
        """Test handler prompt generation without context."""
        prompt = manager.generate_handler_prompt(
            path="/users",
            method="POST",
            description="Create a new user"
        )
        assert "/users" in prompt
        assert "POST" in prompt
        assert "Create a new user" in prompt
    
    def test_generate_usecase_prompt(self, manager):
        """Test usecase prompt generation."""
        prompt = manager.generate_usecase_prompt(
            name="CreateUser",
            description="Create a new user in the system",
            context={"extra": "info"}
        )
        assert "CreateUser" in prompt
        assert "Create a new user in the system" in prompt
    
    def test_generate_usecase_prompt_without_context(self, manager):
        """Test usecase prompt generation without context."""
        prompt = manager.generate_usecase_prompt(
            name="CreateUser",
            description="Create a new user in the system"
        )
        assert "CreateUser" in prompt
        assert "Create a new user in the system" in prompt
    
    def test_generate_model_prompt(self, manager):
        """Test model prompt generation."""
        prompt = manager.generate_model_prompt(
            schema="type User struct { ID int }",
            context={"extra": "info"}
        )
        assert "type User struct { ID int }" in prompt
    
    def test_generate_model_prompt_without_context(self, manager):
        """Test model prompt generation without context."""
        prompt = manager.generate_model_prompt(
            schema="type User struct { ID int }"
        )
        assert "type User struct { ID int }" in prompt
    
    def test_generate_review_prompt(self, manager):
        """Test review prompt generation."""
        prompt = manager.generate_review_prompt(
            code="func main() {}",
            context={"extra": "info"}
        )
        assert "func main() {}" in prompt
    
    def test_generate_review_prompt_without_context(self, manager):
        """Test review prompt generation without context."""
        prompt = manager.generate_review_prompt(
            code="func main() {}"
        )
        assert "func main() {}" in prompt 
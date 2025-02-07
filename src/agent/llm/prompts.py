from typing import Dict, Any, Optional
from dataclasses import dataclass
from string import Template

@dataclass
class PromptTemplate:
    """A template for generating prompts."""
    template: str
    required_variables: set[str]
    
    def format(self, **kwargs) -> str:
        """Format the template with the given variables."""
        missing = self.required_variables - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return Template(self.template).safe_substitute(**kwargs)

class PromptLibrary:
    """A collection of prompts for different tasks."""
    
    @property
    def go_handler_template(self) -> PromptTemplate:
        """Template for generating Go API handlers."""
        return PromptTemplate(
            template="""
Generate a Go API handler for the following endpoint:
Path: ${path}
Method: ${method}
Description: ${description}

Requirements:
- Follow Go best practices and conventions
- Use the Fiber framework
- Implement proper error handling
- Include necessary validation
- Add appropriate comments
- Handle all specified response codes

Additional Context:
${context}
""",
            required_variables={"path", "method", "description"}
        )
    
    @property
    def go_usecase_template(self) -> PromptTemplate:
        """Template for generating Go use cases."""
        return PromptTemplate(
            template="""
Generate a Go use case implementation for the following business logic:
Name: ${name}
Description: ${description}

Requirements:
- Follow clean architecture principles
- Implement proper error handling
- Include necessary validation
- Add appropriate comments
- Make the code testable

Additional Context:
${context}
""",
            required_variables={"name", "description"}
        )
    
    @property
    def go_model_template(self) -> PromptTemplate:
        """Template for generating Go data models."""
        return PromptTemplate(
            template="""
Generate Go data models for the following schema:
${schema}

Requirements:
- Use appropriate Go types
- Include JSON tags
- Add validation tags if needed
- Include comments for each field
- Follow Go naming conventions

Additional Context:
${context}
""",
            required_variables={"schema"}
        )
    
    @property
    def code_review_template(self) -> PromptTemplate:
        """Template for code review prompts."""
        return PromptTemplate(
            template="""
Please review the following Go code:
${code}

Focus on:
- Code correctness
- Best practices
- Error handling
- Performance
- Security
- Testing considerations

Additional Context:
${context}
""",
            required_variables={"code"}
        )

class PromptManager:
    """Manager for handling prompts and their generation."""
    
    def __init__(self):
        self.library = PromptLibrary()
        
    def generate_handler_prompt(
        self,
        path: str,
        method: str,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a prompt for creating a Go API handler."""
        return self.library.go_handler_template.format(
            path=path,
            method=method,
            description=description,
            context=context or {}
        )
    
    def generate_usecase_prompt(
        self,
        name: str,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a prompt for creating a Go use case."""
        return self.library.go_usecase_template.format(
            name=name,
            description=description,
            context=context or {}
        )
    
    def generate_model_prompt(
        self,
        schema: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a prompt for creating Go data models."""
        return self.library.go_model_template.format(
            schema=schema,
            context=context or {}
        )
    
    def generate_review_prompt(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a prompt for code review."""
        return self.library.code_review_template.format(
            code=code,
            context=context or {}
        ) 
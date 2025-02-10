# OpenFinance Agent

[![Tests](https://github.com/cloudwalk/openfinance-agent/actions/workflows/test.yml/badge.svg)](https://github.com/cloudwalk/openfinance-agent/actions/workflows/test.yml)
[![Coverage](./coverage.svg)](./)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

An intelligent agent for generating Go API code from Swagger/OpenAPI specifications. This project uses LLM Compiler and Cognitive Architecture concepts to create a modular and maintainable code generation system.

## Requirements

- Python 3.9 or higher
- OpenAI API key
- Go 1.21 or higher (for generated code)

## Quick Start

```bash
# Install the package
pip install openfinance-agent

# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Generate code from a Swagger spec
python -m openfinance_agent generate --input swagger.yaml --output ./generated
```

## Key Features

- **Intelligent Code Generation**
  - Generates idiomatic Go code from OpenAPI/Swagger specs
  - Uses LLM to understand and implement business logic
  - Follows Go best practices and patterns

- **Cognitive Architecture**
  - Memory Management for context retention
  - Task Planning for complex generations
  - Decision Making for architectural choices
  - Learning System for continuous improvement

- **Developer Experience**
  - Clean, documented code output
  - Customizable templates
  - Extensive configuration options
  - Comprehensive test coverage

## Example

```yaml
# example-api.yaml
openapi: 3.0.0
info:
  title: User Service
  version: 1.0.0
paths:
  /users:
    post:
      summary: Create user
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                name:
                  type: string
                email:
                  type: string
      responses:
        '201':
          description: User created
```

Generate the code:
```bash
openfinance-agent generate -i example-api.yaml -o ./service
```

Generated structure:
```
service/
├── api/
│   ├── handlers/
│   │   └── user_handler.go
│   └── routes.go
├── internal/
│   ├── models/
│   │   └── user.go
│   └── service/
│       └── user_service.go
└── main.go
```

## Architecture

The project follows a modular architecture with the following key components:

### Core Components

1. **Cognitive Architecture**
   - Memory Management
   - Task Planning
   - Decision Making
   - Learning System

2. **LLM Compiler**
   - Code Generation Pipeline
   - AST Transformation
   - Code Optimization
   - Validation System

3. **Agent Components**
   - OpenAI Integration
   - Prompt Management
   - Context Management
   - Code Generation

### Project Structure

```
openfinance-agent/
├── src/
│   ├── cognitive/
│   │   ├── memory.py
│   │   ├── planner.py
│   │   ├── decision.py
│   │   └── learning.py
│   ├── compiler/
│   │   ├── pipeline.py
│   │   ├── ast_transformer.py
│   │   ├── optimizer.py
│   │   └── validator.py
│   ├── agent/
│   │   ├── llm/
│   │   │   ├── base.py
│   │   │   ├── openai.py
│   │   │   └── prompts.py
│   │   ├── context.py
│   │   └── generator.py
│   ├── models/
│   │   ├── swagger.py
│   │   └── golang.py
│   └── utils/
│       ├── logger.py
│       └── config.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── examples/
│   ├── swagger/
│   └── generated/
├── config/
│   └── default.yaml
├── requirements.txt
└── README.md
```

## Features

- Swagger/OpenAPI specification parsing
- Intelligent code generation using LLM
- Modular prompt management
- Extensible LLM provider interface
- Go code generation following best practices
- Cognitive architecture for improved decision making
- Comprehensive test coverage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cloudwalk/openfinance-agent.git
cd openfinance-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

[Usage instructions will be added as the project develops]

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
# Run tests with coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/path/to/test_file.py

# Run tests with specific marker
pytest -m "marker_name"
```

3. Code Style:
```bash
# Format code with black
black .

# Sort imports
isort .

# Run linter
ruff check .

# Run type checker
mypy .
```

4. Pre-commit checks:
```bash
# Run all checks before committing
black . && isort . && ruff check . && mypy . && pytest
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code of Conduct
- Development process
- How to submit pull requests
- Coding standards
- Testing requirements

## Roadmap

- [ ] Support for additional languages (Python, TypeScript)
- [ ] Integration with more LLM providers
- [ ] Enhanced test generation
- [ ] Database schema generation
- [ ] API documentation generation
- [ ] CI/CD pipeline templates

## Support

- [Issue Tracker](https://github.com/cloudwalk/openfinance-agent/issues)
- [Discussions](https://github.com/cloudwalk/openfinance-agent/discussions)
- Email: henrique.menna@gmail.com

## License

This project is licensed under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.en.html). This means you can:
- Use and modify the software freely
- Distribute the software and your modifications
- Must share any modifications under the same license
- Must make source code available when serving over a network

The AGPL-3.0 license ensures that this software remains free and open source, benefiting the entire community. For more details, see the [LICENSE](LICENSE) file. 
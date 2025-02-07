# OpenFinance Agent

An intelligent agent for generating Go API code from Swagger/OpenAPI specifications. This project uses LLM Compiler and Cognitive Architecture concepts to create a modular and maintainable code generation system.

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
git clone https://github.com/yourusername/openfinance-agent.git
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
pytest
```

## Contributing

[Contribution guidelines will be added]

## License

[License information will be added] 
# OpenFinance Agent

![Coverage](./coverage.svg)
[![Tests](https://github.com/henriquemenna/openfinance-agent/actions/workflows/test.yml/badge.svg)](https://github.com/henriquemenna/openfinance-agent/actions/workflows/test.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License Available](https://img.shields.io/badge/License-Commercial-green.svg)](mailto:henrique.menna@gmail.com)

[View Full Coverage Report](https://henriquemenna.github.io/openfinance-agent/coverage/)

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

This project is available under a dual-license model:

### Open Source License (AGPL-3.0)
For non-commercial and open source projects, this software is licensed under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.en.html). This means you can:
- Use the software for non-commercial purposes
- Modify and distribute the software
- Must share any modifications under the same license
- Must make source code available when serving over a network

### Commercial License
For commercial use, a paid license is required. The commercial license includes:
- Full rights to use in commercial projects
- Private modifications allowed
- No requirement to share modifications
- Priority support and updates
- Custom development available

Contact henrique.menna@gmail.com for commercial licensing inquiries.

[![codecov](https://codecov.io/gh/henriquemenna/openfinance-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/henriquemenna/openfinance-agent) 
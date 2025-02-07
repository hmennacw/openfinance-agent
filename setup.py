from setuptools import setup, find_packages

setup(
    name="openfinance-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "jsonschema>=4.0.0",
        "typing-extensions>=4.8.0",
        "python-jose>=3.3.0",
    ],
    python_requires=">=3.9",
) 
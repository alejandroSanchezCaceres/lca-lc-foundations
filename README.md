# 🔗 Introduction to LangChain

## Introduction

Welcome to LangChain Academy’s Introduction to LangChain course!

---

## 🚀 Setup 

### Quick Start Verification

After completing the Setup section, you can run this command to verify your environment:

```bash
uv run python3 env_utils.py
```

### Prerequisites

- The Chrome browser is recommended
- Ensure you're using Python >=3.12, <3.14 [More info](#virtual-environments-and-python)
- [uv](https://docs.astral.sh/uv/) package manager or [pip](https://pypi.org/project/pip/)
- OpenAI API Key, Anthropic API Key, Tavily API Key and Google API Key [More info](#model-providers)

### Installation

Download the course repository

```bash
# Clone the repo
git clone https://github.com/langchain-ai/lca-lc-foundations.git
cd lca-lc-foundations
```

Make a copy of example.env

```bash
# Create .env file
cp example.env .env
```

Edit the .env file to include the keys below. [More info](#environment-variables)

```bash
# Required for model usage
OPENAI_API_KEY='your_openai_api_key_here'
ANTHROPIC_API_KEY='your_anthropic_api_key_here'
TAVILY_API_KEY='your_tavily_api_key_here'
GOOGLE_API_KEY='your_google_api_key_here'

# Optional for evaluation and tracing
LANGSMITH_API_KEY='your_langsmith_api_key_here'
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=lca-lc-foundation
# Uncomment the following if you are on the EU instance:
#LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

Make a virtual environment and install dependencies
```bash
# Create virtual environment and install dependencies
uv sync
```

## 💡 Development Environment

Run Notebooks using either Jupyter or VSCode [More Info](development-environment)

```bash
# Run Jupyter notebooks directly with uv
uv run jupyter lab

# Or activate the virtual environment first, if preferred
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
jupyter lab
```

## 📚 Lessons
This repository contains four sections of notebooks that serve as introductions to many of the most-used features in LangChain.

---

### Section 1: Create Agents

- Foundational models
- Prompting
- Tool calling
- Memory
- multimodal messages
- Project: Personal chef

### Section 2: Advanced Agent Tools

- MCP
- Runtime context (context, ToolRuntime)
- State
- Multi-agent
- Project: Destination wedding planner

### Section 3: Solving Advanced Problems with Middleware

- Summarisation
- HITL
- Dynamic Prompts/Tool Calling/Model Selection
- Project: Email agent

### Section 4: Deployment

- Agent Chat UI
- Deploy on LangSmith, then tie back into Agent Chat UI

## 📖 Related Resources

### Virtual Environments and Python

Managing your Python version is often best done with virtual environments.  This allows you to select a Python version for the course independent of the system Python version. There are many ways to do this. Two popular approaches use `uv` or `pyenv` to select which version to install and use in your virtual environment.

For example, you can select and install the Python version when creating the virtual environment with uv. After you activate the environment, you can proceed with uv sync to install the necessary packages. For additional information, please see [uv](https://docs.astral.sh/uv/).

````bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
````

If you are using pip instead of uv, you may prefer using pyenv to manage your Python versions.  For additional information, please see  [pyenv](https://github.com/pyenv/pyenv).

### Model Providers

If you don’t have an OpenAI API key, you can sign up [here](https://openai.com/index/openai-api/).
You may also obtain additional API keys for [anthropic](https://console.anthropic.com), [tavily](https://tavily.com) or [google](https://docs.langchain.com/oss/python/integrations/providers/google) if needed.

This course has been created using particular model providers.  It is possible to use another provider, but you will need to update the .env file with any required identification (API key), and some code changes will be necessary. See more about this [here](https://docs.langchain.com/oss/python/integrations/providers/all_providers).

### Development Environment

Two popular IDEs that can use notebook files are [Jupyter](jupyter.org/install) and [Cursor](https://cursor.com/download).  Jupyter can be installed locally in the virtual environment with pip if desired.

### Getting Started with LangSmith

- Create a [LangSmith](https://smith.langchain.com/) account
- Create a LangSmith API key

<img width="600" alt="LangSmith Dashboard" src="https://github.com/user-attachments/assets/e39b8364-c3e3-4c75-a287-d9d4685caad5" />

<img width="600" alt="LangSmith API Keys" src="https://github.com/user-attachments/assets/2e916b2d-e3b0-4c59-a178-c5818604b8fe" />

- Update your .env file you created with your new LangSmith API Key.

For more information on LangSmith, see our docs [here](https://docs.langchain.com/oss/python/langchain/studio).

### Environment Variables

This course makes use of the module dotenv, which reads key-value pairs from a .env file rather than relying on preset system environment variables.  

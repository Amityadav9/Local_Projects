# OWL Assistant - Local LLM Task Automation

A powerful multi-agent AI system powered by local LLMs through Ollama, based on the OWL (Optimized Workforce Learning) framework.

This repository provides two interfaces for OWL:
1. **app.py** - A Streamlit web interface
2. **owl_demo.py** - A command-line interface

Both allow you to interact with the same powerful OWL functionality using different modes of operation.

## 🌟 Features

- **Local LLM Integration**: Run advanced AI agents using your local Ollama models
- **Multi-Agent Collaboration**: Leverage multiple specialized agents working together
- **Tool Integration**: Execute code, search the web, analyze data, and process documents
- **No Cloud Required**: Everything runs on your local machine for privacy and control

## 📋 Prerequisites

- Python 3.10+ installed
- [Ollama](https://ollama.ai/) installed and running
- Recommended models:
  - A powerful text model (e.g., qwen2.5:7b)
  - A vision-capable model (e.g., llama3.2-vision:latest)

## 🚀 Quick Start

### Installation

```bash
# Clone the OWL repository
git clone https://github.com/camel-ai/owl.git

# Navigate to project directory
cd owl

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e .

# Install additional requirements for the interfaces
pip install streamlit pandas python-dotenv requests
```

## 🧰 Interface Options

### 1. Streamlit Interface (app.py)

A user-friendly web interface with:
- Demo and custom task tabs
- File upload capabilities
- Model selection
- Output file viewing

**Running the Streamlit Interface:**

```bash
# Start the Streamlit app
streamlit run app.py
```

Then open your browser at http://localhost:8501 to use the web interface.

**Key Notes for Streamlit Version:**
- Browser automation is disabled by default due to Streamlit limitations
- Features a tabbed interface for demo and custom tasks
- Includes file upload capability
- Provides real-time progress updates

### 2. Command Line Interface (owl_demo.py)

A powerful terminal-based interface with:
- Full browser automation support
- Interactive task selection
- Flexible configuration options
- Direct output to terminal

**Running the Command-Line Version:**

```bash
# Run with interactive menu
python owl_demo.py

# Or specify a custom task
python owl_demo.py --task "Write a Python function to calculate Fibonacci numbers"

# Use with specific models
python owl_demo.py --model qwen2.5:7b --vision-model llama3.2-vision:latest
```

**Key Notes for Command-Line Version:**
- Provides full browser automation support
- Features robust error handling
- Includes options for headless mode
- Works well for automated scripts

## 📁 Project Structure

```
.
├── app.py              # Streamlit interface
├── owl_demo.py         # Command-line interface
└── output/             # Output files and results (created automatically)
```

## 🤝 Contributing

Contributions to improve this implementation are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is built on top of the OWL Framework, which is licensed under the Apache License 2.0.

## 🙏 Acknowledgments

Based on the [OWL Framework](https://github.com/camel-ai/owl) by CAMEL-AI.org.

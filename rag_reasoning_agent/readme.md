# 🐋 Deepseek Local RAG Reasoning Agent

## Overview

This Streamlit application provides a powerful local Retrieval-Augmented Generation (RAG) reasoning agent using Ollama models. The app allows you to:
- Chat with multiple local AI models
- Upload PDFs and web pages for context
- Perform web searches
- Configure RAG settings dynamically

## 🚀 Features

### AI Model Support
- Multiple local models supported:
  - deepseek-r1:7b (High capability)
  - qwen2.5-coder (Specialized for coding)
  - llama3.2:3b (Lightweight)
  - qwen2.5:7b (Well-rounded)

### RAG Capabilities
- Document ingestion from PDFs and web URLs
- Qdrant vector database integration
- Configurable similarity thresholds
- Web search fallback and integration

### Flexible Configuration
- GPU acceleration toggle
- Local and cloud Qdrant database support
- Web search mode selection
- Model selection

## 🛠️ Prerequisites

- Python 3.8+
- Ollama
- Docker (optional, for Qdrant)
- CUDA (optional, for GPU acceleration)

## 📦 Installation

1. Clone the repository:
```bash
git clone 
cd deepseek-rag-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pull Ollama models:
```bash
ollama pull deepseek-r1:7b
ollama pull qwen2.5-coder
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
```

## 🖥️ Running the Application

```bash
streamlit run app.py
```

### GPU Acceleration

For CUDA GPU optimization:
```bash
OLLAMA_USE_CUDA=1 ollama serve
```

## 🔧 Configuration Options

### Sidebar Configuration
- **Hardware**: Toggle GPU acceleration
- **Model Selection**: Choose from 4 local models
- **RAG Mode**: Enable/disable Retrieval-Augmented Generation
- **Vector Database**: Local or cloud Qdrant configuration
- **Web Search**: Control web search integration

## 📚 Usage

1. Upload PDFs or enter web URLs in the sidebar
2. Configure your preferred model and settings
3. Start chatting in the main interface
4. Use the web search toggle for additional context

## 🌐 Web Search Modes
- **Disabled**: No web searching
- **Fallback Only**: Search when no documents found
- **Always Available**: Web search can be used anytime

## 🔍 Document Relevance

Adjust the similarity threshold to control the relevance of retrieved documents. Lower values return more documents but might reduce precision.

## 🛡️ Security Notes
- Secure your Qdrant API keys
- Be cautious with web search and document uploads

## 🤝 Contributing
Contributions are welcome! Please submit pull requests or open issues.

## 📄 License
[Insert your license here]

## 🙏 Acknowledgements
- Ollama
- Streamlit
- Qdrant
- Langchain

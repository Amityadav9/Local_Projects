# Enhanced Local AI Chat 🤖

An advanced Streamlit-based chat interface for Ollama models with support for Hugging Face GGUF models, multiple personalities, and customizable system prompts.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Types](#model-types)
- [Personalities](#personalities)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Tips and Best Practices](#tips-and-best-practices)

## Features

### Core Features
- 🤖 Support for standard Ollama models
- 🤗 Integration with Hugging Face GGUF models
- 🎭 Multiple AI personalities
- ⚙️ Customizable system prompts
- 💾 Chat history management
- 🔄 Real-time streaming responses
- 📊 Chat statistics

### Model Support
- Standard Ollama models (Llama, Gemma, Mistral, etc.)
- Hugging Face GGUF models
- Custom quantization options
- Model size categorization

### Interface Features
- Clean, intuitive Streamlit interface
- Advanced settings panel
- Error handling and troubleshooting guides
- Export and save functionality

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running
3. Sufficient system resources for AI models

Required Python packages:
```bash
streamlit
ollama
```

## Installation

1. Clone the repository or download the code:
```bash
git clone <repository-url>
cd enhanced-local-ai-chat
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is installed and running:
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve
```

## Usage

1. Start the application:
```bash
streamlit run chat_app.py
```

2. Access the interface:
- Open your browser
- Navigate to `http://localhost:8501`

### Basic Usage Steps:
1. Select a model source (Standard or Hugging Face)
2. Choose a model
3. Select a personality
4. Start chatting!

## Model Types

### Standard Ollama Models
- llama3.1:latest (4.7GB)
- llama3.2:latest (2.0GB)
- gemma2:9b (5.4GB)
- mistral:7b (4.1GB)
- qwen2.5-coder:latest (4.7GB)
- nemotron-mini:4b (2.7GB)
- phi3:latest (2.2GB)
- nomic-embed-text:latest (274MB)

### Hugging Face Models
Format: `hf.co/username/repository[:quantization]`

Example models:
```bash
hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
hf.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF
hf.co/arcee-ai/SuperNova-Medius-GGUF
```

Quantization options:
- Q4_K_M (default)
- Q8_0 (high quality)
- IQ3_M (small size)
- Q5_K_M
- Q6_K

## Personalities

1. **Balanced**
   - General-purpose responses
   - Well-rounded communication
   - Suitable for most queries

2. **Technical Expert**
   - Detailed technical explanations
   - Code examples and best practices
   - Performance considerations

3. **Creative Writer**
   - Engaging and expressive responses
   - Rich metaphors and analogies
   - Clear and accessible writing

4. **Educational Tutor**
   - Step-by-step explanations
   - Socratic method approach
   - Real-world examples

## Advanced Features

### System Prompts
- Customize assistant behavior
- Edit in real-time
- Save and reload prompts

### Chat Management
- Save chat history
- Export conversations
- Clear chat history
- View chat statistics

### Advanced Settings
- Temperature control
- Max token adjustment
- Custom quantization
- Model-specific optimizations

## Troubleshooting

### Common Issues and Solutions

1. **Model Not Found**
   - Check model path spelling
   - Verify GGUF files exist
   - Try default quantization

2. **Connection Errors**
   - Ensure Ollama is running (`ollama serve`)
   - Check internet connection
   - Verify Hugging Face service status

3. **Resource Issues**
   - Try smaller models
   - Use lighter quantization
   - Close unnecessary applications

4. **Performance Issues**
   - Adjust max tokens
   - Modify temperature setting
   - Consider model size vs. system capabilities

## Tips and Best Practices

1. **Model Selection**
   - Choose models based on your system's capabilities
   - Start with smaller models for testing
   - Use appropriate quantization for your needs

2. **Performance Optimization**
   - Lower temperature for focused responses
   - Adjust max tokens based on needs
   - Use Q4_K_M quantization for balanced performance

3. **Chat Management**
   - Regular export of important conversations
   - Clear chat history for better performance
   - Use appropriate personality for your use case

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]

## Acknowledgments

- Ollama team for the base functionality
- Hugging Face for model accessibility
- Streamlit for the UI framework

---

Built with ❤️ by AMIT

For questions or support, please open an issue on the repository.

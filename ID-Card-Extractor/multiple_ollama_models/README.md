
# ID Document Information Extractor

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.20%2B-red)
![Ollama](https://img.shields.io/badge/ollama-0.1.0%2B-green)

A Streamlit-based application for extracting personal information from ID documents using local LLMs (Large Language Models) powered by Ollama. This application leverages multimodal vision capabilities of models like Gemma 3 and Llama 3 to perform OCR-like extraction without sending your sensitive data to external APIs.

![App Screenshot](docs/app_screenshot.png)

## Features

- 📷 Upload ID documents directly from your computer
- 🔄 Choose between different LLM models (Gemma and Llama families)
- 🌐 Support for both German and English prompts
- 🔍 Extract personal details including:
  - First and last names
  - Date of birth
  - Place of birth
  - Address
  - Expiry date
  - Nationality
- 📊 View extracted data in a structured format
- 💾 Download results as JSON
- 🛡️ Privacy-focused: All processing happens locally on your machine

## Requirements

- Python 3.8 or higher
- Streamlit
- Ollama with vision-capable models installed
- PIL/Pillow (Python Imaging Library)
- Requests

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/id-document-extractor.git
   cd id-document-extractor
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   or install directly:
   ```bash
   pip install streamlit pillow requests
   ```

3. Install and run Ollama:
   - Follow the instructions at [Ollama's official website](https://ollama.com/download)
   - Start the Ollama service:
     ```bash
     ollama serve
     ```

4. Pull the required models:
   ```bash
   # For Gemma models
   ollama pull gemma3:4b
   ollama pull gemma3:12b
   # For Llama vision models
   ollama pull llama3.2-vision:latest
   ollama pull llama3:11b-vision
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run id_extractor_app.py
   ```

2. Access the application in your web browser (typically at http://localhost:8501)

3. Upload an ID document image

4. Select your preferred model and prompt language from the sidebar

5. Click "Extract Information" to process the image

6. View the extracted information and download the JSON if needed

## How It Works

1. The application encodes the uploaded image to base64
2. It sends a request to the local Ollama API with the encoded image and a prompt
3. The LLM processes the image and extracts the requested information
4. The application parses the JSON response and displays it in a structured format

## Model Selection Guide

### Gemma Models
- **gemma3:1b** - Fastest, but less accurate
- **gemma3:4b** - Good balance of speed and accuracy
- **gemma3:12b** - More accurate, but slower
- **gemma3:27b** - Most accurate, slowest processing

### Llama Models
- **llama3.2-vision:latest** - Latest version optimized for vision tasks
- **llama3:11b-vision** - Alternative vision-capable model

## Privacy Notice

This application processes all data locally on your machine. No data is sent to external servers beyond your local Ollama instance. However, please be mindful of the following:

- Do not upload highly sensitive documents unless necessary
- If sharing the extracted data, ensure you've removed any sensitive information
- Consider using test or sample documents when developing or extending this application

## Troubleshooting

- **Model not found** - Ensure you've pulled the required models using Ollama
- **Connection error** - Make sure Ollama is running on your machine
- **Memory issues** - Larger models require significant RAM and VRAM. Try using a smaller model variant

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Ollama team for providing an easy way to run LLMs locally
- Streamlit for the intuitive web app framework
- Google and Meta for releasing Gemma and Llama models respectively

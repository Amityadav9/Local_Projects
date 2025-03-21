# ID Card Information Extractor

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)

A powerful tool that extracts structured information from identity documents including passports, ID cards, residence permits, and more. This application combines OCR, text processing, and optional AI vision analysis to accurately parse document details and save them in JSON format.


## 🌟 Features

- **Multi-format Support**: Process PDF, Image (JPG, PNG), DOCX, and TXT files
- **Dual Extraction Approach**: Combines OCR and AI-powered vision analysis
- **Multilingual Extraction**: Special handling for German/English mixed documents
- **Comprehensive Information Extraction**:
  - Document type identification
  - Personal details (name, nationality, gender, birth date, etc.)
  - Document specifics (ID numbers, issue/expiry dates)
  - Issuing authority and additional information
- **User-friendly Streamlit Interface**: Clean visualization of extracted data
- **Automatic JSON Storage**: Saves structured data with timestamps
- **Intelligent Error Handling**: Graceful fallbacks if extraction methods fail

## 📋 Requirements

- Python 3.8+
- Tesseract OCR
- Ollama with llama3.2:3b model
- Various Python packages (see installation)

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Amityadav9/Local_Projects.git
   cd Local_Projects
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - **Windows**: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **Mac**: `brew install tesseract`

4. Install and run Ollama:
   - Follow instructions at [Ollama](https://ollama.ai/)
   - Pull the required model: `ollama pull llama3.2:3b`

5. Create required directories:
   ```bash
   mkdir -p extracted_id_data
   ```

## 📝 Usage

1. Start the Streamlit application:
   ```bash
   streamlit run id_card_extractor.py
   ```

2. Open the URL displayed in your browser (typically http://localhost:8501)

3. Using the application:
   - Select document type from the dropdown (PDF, Image, DOCX, TXT)
   - Upload your ID document
   - Toggle "Use vision analysis" checkbox if you want to use the vision model
   - View the extracted information
   - Download the JSON file with the extracted data

## 🔧 Configuration

You can modify the following aspects of the extractor:

- **Tesseract Path**: Adjust the path in the code if your Tesseract installation is in a non-standard location
- **Ollama Models**: Change the model names if you want to use different Ollama models
- **Output Directory**: Change the `OUTPUT_DIR` variable to store JSON files elsewhere

## 🧩 How It Works

1. **Document Ingestion**: The application accepts various document formats through the Streamlit interface
2. **Text Extraction**: 
   - PDFs: Extracts text directly, falls back to OCR if needed
   - Images: Uses Tesseract OCR to extract text
   - DOCX/TXT: Extracts text directly from document structure
3. **Vision Analysis** (optional): 
   - Uses Ollama's vision model for visual document understanding
   - Specifically trained to recognize ID document layouts
4. **Information Extraction**:
   - Combines both text and vision analysis
   - Uses LLM to identify and structure the information
   - Handles multilingual content with special attention to German/English documents
5. **Results Storage**:
   - Saves structured data as JSON with timestamps and source information
   - Displays in user-friendly format in the UI
   - Provides download links for extracted data


## ⚠️ Limitations

- **OCR Quality**: Extraction accuracy depends on document image quality
- **Language Support**: Best results with English and German documents
- **Privacy Concerns**: This tool processes ID documents locally but exercise caution with sensitive data
- **No Verification**: This tool extracts information but does not verify document authenticity

## 🔄 Future Improvements

- Support for additional document types and languages
- Enhanced security features for handling sensitive information
- Batch processing capabilities
- Advanced validation of extracted information
- Customizable extraction fields

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



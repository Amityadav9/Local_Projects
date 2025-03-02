# German Chatbot (Deutscher Chat-Assistent)

A Streamlit-based German language chatbot with a fun, witty personality. The bot engages in entertaining conversations about German culture, stereotypes, and everyday situations with a humorous twist.

## Features

- 🗣️ Natural German language conversations
- 😄 Witty and humorous responses
- 🎯 Context-aware dialogue
- 💻 Streamlit web interface
- 🚀 GPU acceleration support
- 📝 Comprehensive logging
- 🎨 Clean, user-friendly UI

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 12GB+ GPU memory (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Amityadav9/Local_Projects/german-chatbot.git
cd german-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# or
myenv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

3. Start chatting with the bot in German!

## Project Structure

```
german-chatbot/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── .streamlit/        # Streamlit configuration
│   └── config.toml    # Streamlit settings
└── logs/              # Log files directory
```

## Dependencies

- streamlit
- torch
- transformers
- logging

## Configuration

The application can be configured through:
1. `.streamlit/config.toml` for Streamlit settings
2. Environment variables:
   - `STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"` (recommended)

## Performance Notes

- GPU acceleration is automatically enabled when available
- Uses mixed precision (FP16) for efficient memory usage
- Implements memory management for long chat sessions
- Caches model for faster responses

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

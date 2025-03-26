# 🛫 Local LLM Travel Planner

An AI-powered travel planner that generates personalized travel itineraries using locally-hosted large language models through Ollama. This Streamlit app automates the process of researching, planning, and organizing your dream vacation, allowing you to explore exciting destinations with ease - all without relying on cloud-based API services like OpenAI.

## ✨ Features

- Research and discover exciting travel destinations, activities, and accommodations
- Customize your itinerary based on the number of days you want to travel
- Utilize the power of locally-hosted LLMs (like Llama, Gemma, Mistral) to generate personalized travel plans
- No OpenAI API key required - works with Ollama on your local machine
- Web search functionality using SerpAPI
- Clean, user-friendly Streamlit interface

## 🚀 Getting Started

### Prerequisites

1. [Ollama](https://ollama.ai/) installed and running on your machine
2. Python 3.7+
3. SerpAPI key for web search functionality

### Installation

1. Clone this repository
```bash
git clone https://github.com/Amityadav9/Local_Projects.git
cd Local_Projects
```

2. Install the required dependencies
```bash
pip install -r requirements.txt
```

3. Make sure you have at least one model pulled in Ollama
```bash
ollama pull llama3.2:3b  # or any other model you prefer
```

### Usage

1. Run the Streamlit app
```bash
streamlit run app.py
```

2. Enter your SerpAPI key when prompted
3. Specify which Ollama model you want to use (e.g., "llama3.2:3b", "gemma3:12b")
4. Enter your desired travel destination and the number of days
5. Click "Generate Itinerary" and watch as the app researches and creates a personalized travel plan!

## 📋 Requirements

```
streamlit
agno
google-search-results
```

## 🧠 How It Works

This app uses two AI agents working together:

1. **Researcher Agent**: Generates search terms based on your destination and travel duration, then uses SerpAPI to search the web for relevant activities and accommodations.

2. **Planner Agent**: Takes the research results and creates a personalized itinerary that includes suggested activities and accommodations for each day of your trip.

Both agents use the locally-hosted LLM through Ollama, making this application completely private and free to use aside from the SerpAPI search costs.

## 🔍 Troubleshooting

- **Model not found**: Make sure you've pulled the model you want to use with Ollama (`ollama pull modelname`)
- **Search not working**: Verify your SerpAPI key is correct
- **Slow responses**: Larger models provide better results but may be slower - try a smaller model for faster responses

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Agno](https://github.com/agno-ai/agno) for the agent framework
- [Ollama](https://ollama.ai/) for the local LLM hosting
- [SerpAPI](https://serpapi.com/) for web search capabilities
- [Streamlit](https://streamlit.io/) for the web interface

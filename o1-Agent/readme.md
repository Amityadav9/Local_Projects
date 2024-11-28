# NemoTron - Multi-Agent AI Assistant

A Python-based multi-agent system that creates an AI assistant (O1) using Ollama for advanced reasoning and problem-solving.

## Features

- Uses Chain of Thought prompting for better reasoning
- Multiple specialized agents working together
- Step-by-step problem analysis
- Automatic saving of conversations and results

## Requirements

- Python 3.x
- Ollama
- Streamlit

## How to Run

1. Make sure you have Ollama installed and running
2. Install the required packages:
   ```
   pip install ollama streamlit
   ```
3. Run the application:
   ```
   python local-o1.py
   ```

## How It Works

1. **CEO Agent**: Creates high-level plans and provides final review
2. **Specialized Agents**: Handle specific steps of the plan
3. **Results**: All conversations and implementations are saved automatically

## File Structure

- `local-o1.py`: Main application file containing all agent logic
- `README.md`: Project documentation
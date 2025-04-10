#!/usr/bin/env python3
# ========= OWL Streamlit Interface =========
# A user-friendly Streamlit interface for interacting with OWL and local LLMs
# 
# Features:
# - Web interface for OWL multi-agent system
# - Demo tasks and custom task entry
# - File upload and processing
# - Model selection for Ollama models
# - Output file viewing and downloading
#
# Usage:
# streamlit run app.py
#
# Note: Browser automation may not work correctly in Streamlit due to event loop 
# conflicts. For reliable browser automation, use owl_demo.py instead.
#
# Author: Your Name
# Based on the OWL Framework by CAMEL-AI.org
# =============================================

import os
import sys
import time
import json
import streamlit as st
import pandas as pd
from datetime import datetime

# Ensure OWL and related dependencies are installed
try:
    from camel.models import ModelFactory
    from camel.toolkits import (
        CodeExecutionToolkit,
        ExcelToolkit,
        ImageAnalysisToolkit,
        SearchToolkit,
        BrowserToolkit,
        FileWriteToolkit,
    )
    from camel.types import ModelPlatformType
    from camel.logger import set_log_level
    from camel.societies import RolePlaying

    try:
        # First try to import from owl
        from owl.utils import run_society
    except ImportError:
        # If that fails, define a simple version
        def run_society(society):
            # Initialize conversation
            society.init_chat()
            # Run the conversation until it's done
            chat_history = society.run_chat()
            # Get the last message from the assistant
            answer = (
                chat_history[-1]["content"] if chat_history else "No answer generated."
            )
            # Return the result
            token_count = {"total": 0}  # Simplified token count
            return answer, chat_history, token_count
except ImportError:
    st.error("""
    ⚠️ Required dependencies not found. Make sure you have installed OWL correctly.
    
    Please follow the installation instructions from the documentation:
    1. Clone the OWL repository
    2. Install the dependencies with `pip install -e .` (or other method from docs)
    
    After installation, run this app again.
    """)
    st.stop()

# Set up logging level
set_log_level(level="INFO")

# Page config
st.set_page_config(
    page_title="OWL Assistant",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("🦉 OWL Assistant")
st.markdown(
    "Interact with OWL - an advanced multi-agent AI system powered by local LLMs."
)

# Initialize session state variables for maintaining state between reruns
if "history" not in st.session_state:
    st.session_state.history = []
if "current_response" not in st.session_state:
    st.session_state.current_response = ""
if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []
if "task_running" not in st.session_state:
    st.session_state.task_running = False
if "history_file" not in st.session_state:
    st.session_state.history_file = os.path.join("output", "conversation_history.json")
if "use_demo" not in st.session_state:
    st.session_state.use_demo = None
if "custom_task" not in st.session_state:
    st.session_state.custom_task = None
if "use_custom" not in st.session_state:
    st.session_state.use_custom = False

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)


# Function to get available Ollama models from the API
def get_ollama_models(ollama_url):
    import requests

    try:
        response = requests.get(f"{ollama_url.rstrip('/v1')}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            return []
    except Exception as e:
        st.warning(f"⚠️ Could not fetch models from Ollama: {e}")
        return []


# Function to construct OWL society with agent configuration
def construct_society(question, main_model, vision_model, ollama_url, headless=True, use_browser=False):
    """Construct a society of agents based on the given question.
    
    Args:
        question: The task or question to process
        main_model: Primary text model name
        vision_model: Vision-capable model name
        ollama_url: URL to the Ollama API
        headless: Whether to run browser in headless mode
        use_browser: Whether to enable browser automation
        
    Returns:
        RolePlaying: Configured society of agents
    """
    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=main_model,
            url=ollama_url,
            model_config_dict={"temperature": 0.2, "max_tokens": 1000000},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=main_model,
            url=ollama_url,
            model_config_dict={"temperature": 0.2, "max_tokens": 1000000},
        ),
        "browsing": ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=vision_model,
            url=ollama_url,
            model_config_dict={"temperature": 0.4, "max_tokens": 1000000},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=main_model,
            url=ollama_url,
            model_config_dict={"temperature": 0.4, "max_tokens": 1000000},
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type=vision_model,
            url=ollama_url,
            model_config_dict={"temperature": 0.4, "max_tokens": 1000000},
        ),
    }

    # Configure toolkits based on the task
    tools = []
    
    # Add browser tools only if requested
    if use_browser:
        try:
            browser_tools = BrowserToolkit(
                headless=headless,
                web_agent_model=models["browsing"],
                planning_agent_model=models["planning"],
            ).get_tools()
            tools.extend(browser_tools)
        except Exception as e:
            st.warning(f"Browser automation unavailable: {str(e)}")
            st.info("Continuing without browser automation capabilities.")
    
    # Add other tools that don't rely on browser automation
    tools.extend([
        # Code execution for data analysis, generation, etc.
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        
        # Image analysis
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        
        # Search capabilities
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_wiki,
        
        # Excel processing
        *ExcelToolkit().get_tools(),
        
        # File operations
        *FileWriteToolkit(output_dir="./output").get_tools(),
    ])

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = RolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    return society


# Function to save conversation history to file
def save_history():
    try:
        with open(st.session_state.history_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save history: {e}")


# Function to load conversation history from file
def load_history():
    try:
        if os.path.exists(st.session_state.history_file):
            with open(st.session_state.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load history: {e}")
    return []


# Function to use demo task
def use_demo():
    if st.session_state.selected_demo != "Select a demo...":
        st.session_state.use_demo = st.session_state.selected_demo


# Function to use custom task
def use_custom():
    st.session_state.use_custom = True


# Demo task examples
def get_demo_tasks():
    return [
        "Create a simple data visualization of the following numbers: 10, 25, 15, 30, 20. Use a bar chart.",
        "Find the current weather in New York City and create a simple report.",
        "Write a simple Python program that generates the Fibonacci sequence up to the 10th number.",
        "Create an example CSV file with 5 rows of sample customer data and then read it back to display statistics.",
        "Using search, find the top 3 most popular programming languages in 2024 and summarize their key features.",
        "Analyze the file 'example.csv' if it exists in the current directory and provide statistics.",
    ]


# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Ollama server URL
    ollama_url = st.text_input(
        "Ollama Server URL",
        value="http://localhost:11434/v1",
        help="URL to your local Ollama API",
    )

    # Refresh models button
    if st.button("Refresh Models"):
        with st.spinner("Fetching models..."):
            st.session_state.ollama_models = get_ollama_models(ollama_url)

    # If models haven't been fetched yet, get them
    if not st.session_state.ollama_models:
        with st.spinner("Fetching models..."):
            st.session_state.ollama_models = get_ollama_models(ollama_url)

    # Model selection
    if st.session_state.ollama_models:
        # Main model selection
        default_main_model = (
            "qwen2.5:7b"
            if "qwen2.5:7b" in st.session_state.ollama_models
            else st.session_state.ollama_models[0]
        )
        main_model = st.selectbox(
            "Main Model",
            st.session_state.ollama_models,
            index=st.session_state.ollama_models.index(default_main_model)
            if default_main_model in st.session_state.ollama_models
            else 0,
            help="Model for text understanding and generation",
        )

        # Vision model selection
        vision_models = [
            model
            for model in st.session_state.ollama_models
            if any(
                name in model.lower()
                for name in ["llava", "llama3.2-vision", "gemma3", "clip"]
            )
        ]
        default_vision_model = (
            "llama3.2-vision:latest"
            if "llama3.2-vision:latest" in vision_models
            else (vision_models[0] if vision_models else main_model)
        )
        vision_model = st.selectbox(
            "Vision Model",
            vision_models if vision_models else st.session_state.ollama_models,
            index=vision_models.index(default_vision_model)
            if default_vision_model in vision_models and vision_models
            else 0,
            help="Model for image and visual understanding",
        )
    else:
        st.warning("⚠️ No Ollama models found. Please check your Ollama server.")
        main_model = "qwen2.5:7b"  # Default fallback
        vision_model = "llama3.2-vision:latest"  # Default fallback

    # Browser option toggle (default off to avoid errors)
    use_browser = st.toggle(
        "Enable Browser Automation",
        value=False,
        help="Warning: May cause errors in Streamlit. For browser automation, use the command-line version.",
    )
    
    # Headless mode toggle (only visible if browser is enabled)
    headless = True
    if use_browser:
        headless = st.toggle(
            "Headless Browser Mode",
            value=True,
            help="Run browser without visible window (more efficient)",
        )
        st.warning("Browser automation may not work in Streamlit due to event loop conflicts. If you encounter errors, disable it.")

    # Advanced settings expander
    with st.expander("Advanced Settings"):
        # History file path
        history_file = st.text_input(
            "History File",
            value=st.session_state.history_file,
            help="Path to save conversation history",
        )
        st.session_state.history_file = history_file

        # Clear history button
        if st.button("Clear History"):
            st.session_state.history = []
            save_history()
            st.success("✅ History cleared!")

# Task selection tabs
tab1, tab2 = st.tabs(["Demo Tasks", "Custom Task"])

# Tab 1: Demo Tasks
with tab1:
    st.header("Demo Tasks")
    demo_tasks = get_demo_tasks()
    # Using session state to avoid rerun issues
    st.selectbox(
        "Select a demo task", 
        ["Select a demo..."] + demo_tasks,
        key="selected_demo"
    )

    if st.button("Run Demo Task"):
        use_demo()

# Tab 2: Custom Task
with tab2:
    st.header("Custom Task")
    
    # Custom task text area
    custom_task = st.text_area(
        "Enter your custom task:",
        height=100,
        placeholder="Example: Create a Python function that calculates the factorial of a number and explain how it works.",
        key="custom_task_input"
    )
    
    # File upload option
    uploaded_file = st.file_uploader("Optionally upload a file to analyze:", type=["csv", "txt", "json", "xlsx", "docx", "pdf"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = os.path.join("output", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved to {file_path}")
        
        # Update custom task to include the file
        if custom_task:
            custom_task += f"\n\nUse the uploaded file: {file_path}"
        else:
            custom_task = f"Analyze the uploaded file: {file_path}"
    
    if st.button("Run Custom Task"):
        if custom_task:
            st.session_state.custom_task = custom_task
            st.session_state.use_custom = True
        else:
            st.error("Please enter a custom task first.")

# Main chat area
st.header("Chat")

# Load previous history if available
if not st.session_state.history:
    st.session_state.history = load_history()

# Display chat history
for item in st.session_state.history:
    if item["role"] == "user":
        with st.chat_message("user"):
            st.markdown(item["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(item["content"])

# Task input - check for demo selection or custom task
task_input = None
if st.session_state.use_demo:
    task_input = st.session_state.use_demo
    # Clear it so it's only used once
    st.session_state.use_demo = None
elif st.session_state.use_custom:
    task_input = st.session_state.custom_task
    # Clear it so it's only used once
    st.session_state.use_custom = False
else:
    task_input = st.chat_input(
        "Or enter your task or question here...",
        disabled=st.session_state.task_running,
    )

# Process the task
if task_input:
    # Add user message to history and display
    with st.chat_message("user"):
        st.markdown(task_input)

    # Add to history
    st.session_state.history.append({"role": "user", "content": task_input})

    # Set flag to indicate task is running
    st.session_state.task_running = True

    # Create assistant message placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Thinking...")

        try:
            # Create progress
            progress_bar = st.progress(0)

            # Run in steps to show progress
            progress_bar.progress(10, text="Initializing the society...")
            time.sleep(0.5)

            # Construct society with browser option
            society = construct_society(
                task_input,
                main_model,
                vision_model,
                ollama_url,
                headless,
                use_browser=use_browser  # Pass the browser toggle value
            )

            progress_bar.progress(30, text="Analyzing the task...")
            time.sleep(0.5)

            progress_bar.progress(50, text="Working on a solution...")

            # Run society
            start_time = time.time()
            answer, chat_history, token_count = run_society(society)
            end_time = time.time()

            progress_bar.progress(90, text="Finalizing the response...")
            time.sleep(0.5)

            # Display the final answer
            message_placeholder.markdown(answer)
            progress_bar.progress(100, text="Done!")

            # Add to history
            st.session_state.history.append({"role": "assistant", "content": answer})

            # Save history
            save_history()

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Processing Time", f"{(end_time - start_time):.2f} seconds")
            with col2:
                total_tokens = token_count.get("total", 0)
                st.metric(
                    "Total Tokens", f"{total_tokens:,}" if total_tokens else "N/A"
                )

        except Exception as e:
            error_message = f"⚠️ Error: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.history.append(
                {"role": "assistant", "content": error_message}
            )
            st.error(f"Details: {str(e)}")

        finally:
            # Reset the running flag
            st.session_state.task_running = False

# File viewer section
st.header("Output Files")
output_dir = "output"
if os.path.exists(output_dir):
    output_files = [
        f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))
    ]

    if output_files:
        selected_file = st.selectbox("Select a file to view", output_files)

        if selected_file:
            file_path = os.path.join(output_dir, selected_file)
            file_extension = os.path.splitext(selected_file)[1].lower()

            # Display file based on its type
            if file_extension in [
                ".txt",
                ".md",
                ".csv",
                ".json",
                ".py",
                ".js",
                ".html",
                ".css",
            ]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    if file_extension == ".csv":
                        df = pd.read_csv(file_path)
                        st.dataframe(df)
                    elif file_extension == ".json":
                        try:
                            data = json.loads(content)
                            st.json(data)
                        except:
                            st.text(content)
                    else:
                        st.text(content)

                    # Download button
                    st.download_button(
                        label="Download File",
                        data=content,
                        file_name=selected_file,
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            elif file_extension in [".png", ".jpg", ".jpeg", ".gif"]:
                st.image(file_path)
            else:
                st.warning(f"File preview not supported for {file_extension} files.")
    else:
        st.info("No output files generated yet.")
else:
    st.info("Output directory not found. It will be created when files are generated.")

# Command-line version information
st.header("Command-Line Version")
st.markdown("""
If you need browser automation or encounter issues with this Streamlit interface, 
try the command-line version of OWL which doesn't have the same event loop limitations.

```bash
# Using the standalone script
python owl_demo.py
```

For browser automation specifically, the command-line version is recommended.
""")

# Footer
st.markdown("---")
st.caption("OWL Assistant - Powered by local LLMs through Ollama")

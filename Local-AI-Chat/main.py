import streamlit as st
import time
from ollama import Client
from typing import Dict, List
import json
from datetime import datetime

# System Prompts and Personalities
DEFAULT_SYSTEM_PROMPT = """You are an advanced AI assistant with expertise in a wide range of topics.
Your role is to provide accurate, helpful, and insightful responses to user queries. Please follow these guidelines:

1. Provide comprehensive and well-structured answers.
2. Use examples and analogies to explain complex concepts when appropriate.
3. If a query is ambiguous, ask for clarification before answering.
4. Always prioritize factual accuracy and cite sources when possible.
5. Be respectful and considerate in your responses.
6. If you're unsure about something, acknowledge your limitations and suggest where the user might find more information.
7. Tailor your language and explanations to the user's apparent level of understanding.

Your goal is to assist users effectively while promoting learning and critical thinking."""

PERSONALITY_PROMPTS = {
    "Balanced": DEFAULT_SYSTEM_PROMPT,
    "Technical Expert": """You are a technical expert AI assistant with deep knowledge of programming, software development, and computer science.
Focus on providing detailed technical explanations, code examples, and best practices. When discussing code:
1. Always explain the logic behind solutions
2. Suggest optimizations and improvements
3. Point out potential pitfalls
4. Include comments in code examples
5. Consider performance implications""",
    "Creative Writer": """You are a creative and engaging AI assistant with a flair for expression.
Your responses should be:
1. Engaging and well-crafted
2. Rich with metaphors and analogies
3. Clear and accessible
4. Imaginative yet precise
5. Structured for easy reading""",
    "Educational Tutor": """You are an educational AI tutor focused on breaking down complex topics into understandable pieces.
Your approach should:
1. Use the Socratic method when appropriate
2. Provide step-by-step explanations
3. Use real-world examples
4. Check understanding frequently
5. Encourage critical thinking""",
}


class ChatApp:
    def __init__(self):
        self.client = Client()
        self.setup_page()
        self.initialize_session_state()
        self.setup_sidebar()
        self.render_chat_interface()

    def setup_page(self):
        st.set_page_config(
            page_title="🤖 Enhanced Local AI Chat",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.title("🤖 Enhanced Local AI Chat")

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "model" not in st.session_state:
            st.session_state.model = "llama3.2:latest"
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7
        if "max_tokens" not in st.session_state:
            st.session_state.max_tokens = 2000
        if "custom_model" not in st.session_state:
            st.session_state.custom_model = ""
        if "model_source" not in st.session_state:
            st.session_state.model_source = "Standard"
        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        if "personality" not in st.session_state:
            st.session_state.personality = "Balanced"
        if "custom_prompt" not in st.session_state:
            st.session_state.custom_prompt = ""

    def setup_sidebar(self):
        with st.sidebar:
            st.title("⚙️ Chat Settings")

            # Personality and System Prompt Settings
            st.markdown("### 🎭 Assistant Personality")
            personality = st.selectbox(
                "Select Personality",
                list(PERSONALITY_PROMPTS.keys()),
                help="Choose the assistant's communication style",
            )

            if personality != st.session_state.personality:
                st.session_state.personality = personality
                st.session_state.system_prompt = PERSONALITY_PROMPTS[personality]

            with st.expander("✏️ Custom System Prompt"):
                custom_prompt = st.text_area(
                    "Edit System Prompt",
                    value=st.session_state.system_prompt,
                    height=300,
                    help="Customize how the AI assistant behaves",
                )
                if custom_prompt != st.session_state.system_prompt:
                    st.session_state.system_prompt = custom_prompt
                    if st.session_state.messages:
                        if st.button("Apply New Prompt"):
                            st.session_state.messages = []
                            st.experimental_rerun()

            # Model Selection
            st.markdown("### 🤖 Model Selection")
            st.session_state.model_source = st.radio(
                "Select Model Source",
                ["Standard Models", "Hugging Face Models"],
                help="Choose between pre-packaged models or custom Hugging Face models",
            )

            if st.session_state.model_source == "Standard Models":
                # Standard model selection
                standard_models = {
                    "llama3.1:latest": "Latest Llama 3.1 model (4.7GB)",
                    "llama3.2:latest": "Optimized Llama 3.2 model (2.0GB)",
                    "gemma2:9b": "Google's Gemma 2 9B model (5.4GB)",
                    "mistral:7b": "Mistral 7B model (4.1GB)",
                    "qwen2.5-coder:latest": "Qwen 2.5 Coder (4.7GB)",
                    "nemotron-mini:4b": "Nemotron Mini 4B (2.7GB)",
                    "phi3:latest": "Microsoft Phi-3 (2.2GB)",
                }

                st.session_state.model = st.selectbox(
                    "Select Model",
                    list(standard_models.keys()),
                    help="Choose a pre-packaged model",
                )
                st.info(f"📝 {standard_models[st.session_state.model]}")

            else:
                # Hugging Face model input
                st.markdown("### 🤗 Hugging Face Model")
                st.markdown("""
                Example format: `hf.co/username/model-name:latest`  
                or with quantization: `hf.co/username/model-name:Q8_0`
                """)

                # Template selection for common models
                hf_templates = {
                    "": "Custom Input",
                    "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest": "Llama 3.2 1B Instruct",
                    "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0": "Llama 3.2 3B Instruct Q8",
                    "hf.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF:Q8_0": "Mistral 7B Instruct Q8",
                    "hf.co/TheBloke/neural-chat-7B-v3-2-GGUF:Q6_K": "Neural Chat 7B Q6",
                }

                template = st.selectbox(
                    "Quick Templates",
                    list(hf_templates.keys()),
                    format_func=lambda x: hf_templates[x],
                    help="Choose a pre-configured model or enter your own",
                )

                custom_input = st.text_input(
                    "Custom Hugging Face Model",
                    value=template,
                    help="Enter the full Hugging Face model path",
                )

                if custom_input:
                    st.session_state.model = custom_input

                st.info("""
                💡 **Tips:**
                - Models must be in GGUF format
                - Q8_0 is balanced for quality/speed
                - Q4_K is faster but lower quality
                - Use :latest for the default quantization
                """)

            # Advanced settings
            with st.expander("🛠️ Advanced Settings"):
                st.session_state.temperature = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Higher = more creative, Lower = more focused",
                )

                st.session_state.max_tokens = st.slider(
                    "Max Response Length",
                    min_value=500,
                    max_value=4000,
                    value=2000,
                    step=500,
                )

            # Chat management
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.experimental_rerun()

            if len(st.session_state.messages) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Chat"):
                        self.save_chat_history()
                with col2:
                    if st.button("📤 Export JSON"):
                        self.export_chat_history()

        # Sidebar information
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📚 Features Guide")
            st.info("""
            **Now Supporting:**
            1. Multiple AI Personalities
            2. Custom System Prompts
            3. Hugging Face Models
            4. Advanced Settings
            
            **Tips for best results:**
            - Choose personality based on your needs
            - Adjust temperature for creativity vs precision
            - Use Q8_0 quantization for best balance
            """)

    def process_user_input(self, prompt: str):
        # Add system prompt if it's the first message
        if not st.session_state.messages:
            st.session_state.messages.append(
                {"role": "system", "content": st.session_state.system_prompt}
            )

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Show loading message with model info
                model_display = (
                    st.session_state.model.split("/")[-1].split(":")[0]
                    if "hf.co" in st.session_state.model
                    else st.session_state.model
                )
                with st.spinner(f"Generating response using {model_display}..."):
                    for response in self.client.chat(
                        model=st.session_state.model,
                        messages=st.session_state.messages,
                        stream=True,
                        options={
                            "temperature": st.session_state.temperature,
                            "max_tokens": st.session_state.max_tokens,
                        },
                    ):
                        full_response += response["message"]["content"]
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("""
                Troubleshooting tips:
                1. Check if the model path is correct
                2. Ensure the model is in GGUF format
                3. Try a different quantization (e.g., :Q8_0)
                4. Check your internet connection
                """)
                return

    def render_chat_interface(self):
        try:
            for message in st.session_state.messages:
                if message["role"] != "system":  # Don't display system prompts
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if prompt := st.chat_input("What's on your mind?"):
                self.process_user_input(prompt)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    def save_chat_history(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "messages": st.session_state.messages,
                        "model": st.session_state.model,
                        "system_prompt": st.session_state.system_prompt,
                        "timestamp": timestamp,
                    },
                    f,
                    indent=2,
                )
            st.sidebar.success(f"Chat saved as {filename}")
        except Exception as e:
            st.sidebar.error(f"Error saving chat: {str(e)}")

    def export_chat_history(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_export_{timestamp}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "messages": [
                            msg
                            for msg in st.session_state.messages
                            if msg["role"] != "system"
                        ],
                        "metadata": {
                            "model": st.session_state.model,
                            "personality": st.session_state.personality,
                            "timestamp": timestamp,
                        },
                    },
                    f,
                    indent=2,
                )
            st.sidebar.success(f"Chat exported as {filename}")
        except Exception as e:
            st.sidebar.error(f"Error exporting chat: {str(e)}")


if __name__ == "__main__":
    chat_app = ChatApp()

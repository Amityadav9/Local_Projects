import os

# Force disable file watching
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import List, Dict
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chatbot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def initialize_model():
    """Initialize model with caching to prevent reloading"""
    try:
        # Clear any existing cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        model_name = "malteos/german-r1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure model loading for RTX 4000
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        logger.info(
            f"Model loaded successfully. GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise


def generate_response(model, tokenizer, user_input: str) -> str:
    """Generate response using the model"""
    try:
        system_prompt = """Du bist ein freundlicher und witziger deutscher Chatbot mit viel Persönlichkeit! 
        Formatiere deine Antworten IMMER genau in diesem Format:

        [REASONING]
        • Hier kommen deine Gedanken in Stichpunkten
        • Sei kreativ und witzig!
        • Füge gerne einen passenden Witz ein

        [ANSWER] 
        Hier kommt deine Hauptantwort. Sei direkt und unterhaltsam!

        [JOKE]
        Hier kommt ein passender Witz oder eine lustige Bemerkung zum Abschluss!
        """

        full_prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"

        with torch.inference_mode():
            inputs = tokenizer(full_prompt, return_tensors="pt")

            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's response
        response = response.split("Assistant:")[-1].strip()
        logger.info("Response generated successfully")
        return response

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "Es tut mir leid, aber ich konnte keine Antwort generieren. Bitte versuchen Sie es erneut."


def main():
    st.set_page_config(page_title="Deutscher Chatbot", page_icon="🤖", layout="wide")

    st.title("Deutscher Chatbot 🤖")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create a sidebar with model information
    with st.sidebar:
        st.markdown("### Model Information")
        if torch.cuda.is_available():
            st.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("Running on CPU")

    # Initialize model
    try:
        with st.spinner("Lade das Modell..."):
            model, tokenizer = initialize_model()
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {str(e)}")
        return

    # Create two columns for a better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ihre Nachricht"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Denke nach..."):
                    response = generate_response(model, tokenizer, prompt)
                    # Format the response for better readability
                    formatted_response = response.split("Assistant:")[-1].strip()

                    # Replace section markers with formatted headers
                    formatted_response = formatted_response.replace(
                        "[REASONING]", "### 🤔 Gedankengang"
                    )
                    formatted_response = formatted_response.replace(
                        "[ANSWER]", "### 💡 Antwort"
                    )
                    formatted_response = formatted_response.replace(
                        "[JOKE]", "### 😄 Zum Schluss"
                    )

                    # Add some spacing and styling
                    formatted_response = formatted_response.replace("•", "\n•")

                    st.markdown(formatted_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

    with col2:
        st.markdown("### Hinweise")
        st.markdown("""
        - Stellen Sie Ihre Fragen auf Deutsch
        - Der Bot antwortet mit Begründung und Antwort
        - Die Antworten werden durch KI generiert
        """)


if __name__ == "__main__":
    main()

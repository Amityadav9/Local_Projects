import streamlit as st
import requests
import json
import base64
import re
from PIL import Image
import io

st.set_page_config(page_title="ID Information Extractor", layout="wide")


def encode_image_to_base64(image_bytes):
    """Convert image bytes to base64 encoding"""
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_id_info(image_bytes, model, prompt_language="German"):
    """Extract information from an ID document using Ollama"""

    # Select prompt based on language
    if prompt_language == "German":
        prompt = "Du bist ein OCR(Optical Character Recognition) Experte. Extrahiere folgende Informationen aus diesem Ausweis: Vornamen, Nachname, Geburtsdatum, Geburtsort, Anschrift, Gültigkeitsdatum und Staatsangehörigkeit. Beachte dass es bei den Vornamen mehrer Namen geben kann - gib bitte alle Vornamen aus. Formatiere die Informationen als JSON."
    else:  # English
        prompt = "You are an OCR (Optical Character Recognition) expert. Extract the following information from this ID document: First names, Last name, Date of birth, Place of birth, Address, Expiry date, and Nationality. Note that there may be multiple first names - please output all first names. Format the information as JSON."

    # Encode the image
    image_base64 = encode_image_to_base64(image_bytes)

    # Create API request payload for Ollama
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    # Add model-specific stop tokens if needed
    if "gemma" in model.lower():
        payload["options"]["stop"] = ["<end_of_turn>"]

    # Send request to Ollama API
    with st.spinner(f"Processing with {model}..."):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate", json=payload, timeout=60
            )

            if response.status_code != 200:
                st.error(f"Error: API returned status code {response.status_code}")
                st.code(response.text)
                return None

            # Extract response
            result = response.json()
            response_text = result.get("response", "")

            # Extract JSON from response using regex
            json_pattern = re.compile(r"{.*}", re.DOTALL)
            match = json_pattern.search(response_text)

            if match:
                try:
                    json_str = match.group(0).strip()
                    data = json.loads(json_str)
                    return data, response_text
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing JSON: {e}")
                    return None, response_text
            else:
                st.warning("No JSON found in response")
                return None, response_text

        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {e}")
            return None, str(e)


def main():
    st.title("ID Information Extractor")
    st.write("Upload an ID document image to extract personal information")

    # Sidebar for model selection
    st.sidebar.header("Model Settings")
    model_type = st.sidebar.radio("Select Model Family", options=["Gemma", "Llama"])

    # Model selection based on family
    if model_type == "Gemma":
        model = st.sidebar.selectbox(
            "Select Gemma Model",
            options=["gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b"],
        )
    else:  # Llama
        model = st.sidebar.selectbox(
            "Select Llama Model",
            options=["llama3.2-vision:latest", "llama3:11b-vision"],
        )

    # Language selection for prompt
    prompt_language = st.sidebar.radio("Prompt Language", options=["German", "English"])

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)

        # Process button
        if st.button("Extract Information"):
            with col2:
                st.subheader("Extracted Information")

                # Process the image with the selected model
                result, raw_output = extract_id_info(
                    image_bytes, model, prompt_language
                )

                if result:
                    # Display the extracted information
                    for key, value in result.items():
                        st.write(f"**{key}**: {value}")

                    # Option to download JSON
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(result, indent=2, ensure_ascii=False),
                        file_name="extracted_id_info.json",
                        mime="application/json",
                    )

                # Display raw model output in an expander
                with st.expander("Raw Model Output"):
                    st.code(raw_output)


if __name__ == "__main__":
    main()


import streamlit as st
import os
import json
import re
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from PIL import Image
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from streamlit.runtime.uploaded_file_manager import UploadedFile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create output directory for JSON files
OUTPUT_DIR = Path("extracted_id_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set Tesseract path for OCR (modify this based on your installation)
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # For Windows
)
# For Linux: pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


def extract_text_from_image(image):
    """Extract text from image using OCR."""
    try:
        text = pytesseract.image_to_string(image, lang="eng+deu")
        return text
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return ""


def process_file(file_data, file_type, file_name):
    """Process different file types to extract ID card information."""
    try:
        extracted_text = ""

        # Process different file types
        if file_type == "PDF":
            if isinstance(file_data, UploadedFile):
                file_bytes = BytesIO(file_data.read())
                # First try to extract text directly
                pdf_reader = PdfReader(file_bytes)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"

                # If little text was extracted, try OCR
                if len(extracted_text.strip()) < 50:
                    logger.info(
                        "PDF text extraction yielded minimal text, trying OCR..."
                    )
                    file_bytes.seek(0)  # Reset file pointer
                    images = convert_from_bytes(file_bytes.read())
                    for image in images:
                        extracted_text += extract_text_from_image(image) + "\n"

        elif file_type == "Image":
            if isinstance(file_data, UploadedFile):
                image = Image.open(BytesIO(file_data.read()))
                extracted_text = extract_text_from_image(image)

        elif file_type == "DOCX":
            if isinstance(file_data, UploadedFile):
                doc = Document(BytesIO(file_data.read()))
                extracted_text = "\n".join([para.text for para in doc.paragraphs])

        elif file_type == "TXT":
            if isinstance(file_data, UploadedFile):
                extracted_text = file_data.read().decode("utf-8")

        # Log extracted text for debugging
        logger.info(
            f"Extracted text from {file_name} ({len(extracted_text)} characters)"
        )
        if len(extracted_text) < 200:
            logger.info(f"Text sample: {extracted_text}")
        else:
            logger.info(f"Text sample: {extracted_text[:200]}...")

        return extracted_text

    except Exception as e:
        logger.error(f"Error processing file {file_name}: {str(e)}")
        return ""


def extract_id_card_info(text, file_name, vision_analysis=None):
    """Extract ID card information using Ollama LLM."""
    try:
        # Initialize Ollama
        llm = Ollama(
            model="llama3.2:3b",
            temperature=0.2,
            base_url="http://localhost:11434",
            timeout=60,
        )

        # Create prompt to extract information
        prompt = f"""
You are an expert at extracting information from ID documents like passports, ID cards, and residence permits.

Analyze the following text extracted from an ID document and extract the information in a structured format.
Be aware that the text might be noisy due to OCR errors.

Document text:
```
{text[:4000]}  # Limit text length for the prompt
```

{"" if vision_analysis is None else "Additional visual analysis:" + vision_analysis}

Extract the following information in valid JSON format:
{{
  "document_type": "The type of document (Passport, ID Card, Residence Permit, etc.)",
  "full_name": "The person's full name",
  "document_number": "The document's identification number",
  "nationality": "The person's nationality (country name or code)",
  "birth_date": "Date of birth in format DD MM YYYY if found",
  "issue_date": "Document issue date if found",
  "expiry_date": "Document expiry date if found", 
  "gender": "Gender (M or F)",
  "birth_place": "Place of birth if found",
  "issuing_authority": "The authority that issued the document",
  "additional_info": "Any other relevant information"
}}

Only return the JSON object, no other text. If information is not found, use null for that field.
"""

        # Get response from LLM
        response = llm.invoke(prompt)

        # Extract JSON
        try:
            # Find JSON in the response
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                json_str = match.group(0)
                extracted_info = json.loads(json_str)

                # Add file source and extraction timestamp
                extracted_info["source_file"] = file_name
                extracted_info["extraction_timestamp"] = datetime.now().isoformat()

                # Save to JSON file
                json_file_name = (
                    OUTPUT_DIR / f"{file_name.split('.')[0]}_extracted.json"
                )
                with open(json_file_name, "w", encoding="utf-8") as f:
                    json.dump(extracted_info, f, indent=2, ensure_ascii=False)

                logger.info(f"Extracted information saved to {json_file_name}")
                return extracted_info
            else:
                logger.error("No JSON found in the response")
                return {
                    "error": "No structured data could be extracted",
                    "raw_response": response,
                }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return {"error": "Failed to parse JSON", "raw_response": response}

    except Exception as e:
        logger.error(f"Error extracting information: {str(e)}")
        return {"error": str(e)}


def perform_vision_analysis(file_data, file_name):
    """Perform vision-based analysis if a vision model is available."""
    try:
        from agno.agent import Agent
        from agno.media import Image as AgnoImage
        from agno.models.ollama import Ollama

        logger.info("Vision model is available, performing visual analysis")

        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file_name}"
        with open(temp_file_path, "wb") as f:
            f.write(file_data.read())

        # Reset file pointer
        file_data.seek(0)

        # Create vision agent
        agent = Agent(
            model=Ollama(id="llama3.2-vision:latest"),
            markdown=True,
        )

        # Vision prompt for ID document analysis
        vision_prompt = """
This appears to be an ID document. Please analyze it carefully and extract the exact information listed below.

Note that the document may contain both German and English text. Look for these German field names and their English equivalents:
- AUFENTHALTSTITEL/AUFENTHALTSERLAUBNIS = Residence Permit
- PERSONALAUSWEIS = ID Card
- REISEPASS = Passport
- NAME/NAMEN/NACHNAME = Surname/Last name
- VORNAME(N) = First/Given name(s)
- GEBURTSDATUM = Date of birth
- GEBURTSORT = Place of birth
- STAATSANGEHÖRIGKEIT = Nationality
- GESCHLECHT = Sex/Gender
- AUSSTELLUNGSDATUM = Date of issue
- KARTE GÜLTIG BIS = CARD EXPIRY
- BEHÖRDE = Authority/Issuing authority

Please extract and list the following information:
- Document type
- Full name
- Document/ID number
- Nationality
- Birth date
- Issue date
- Expiry date
- Gender
- Birth place/city
- Issuing authority

Provide ONLY the extracted information in a clean list format.
"""

        # Get response from vision model
        response = agent.print_response(
            vision_prompt,
            images=[AgnoImage(filepath=temp_file_path)],
        )

        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return str(response)

    except ImportError:
        logger.warning("Agno vision module not available, skipping vision analysis")
        return None
    except Exception as e:
        logger.error(f"Error in vision analysis: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="ID Card Information Extractor", layout="wide")

    st.title("🆔 ID Card Information Extractor")
    st.markdown("""
    This app extracts information from various ID documents like passports, national ID cards, and residence permits.
    Upload your document, and the app will extract key information and save it as a JSON file.
    """)

    # File uploader
    file_type = st.selectbox("Select document type", ["PDF", "Image", "DOCX", "TXT"])

    file_extensions = {
        "PDF": ["pdf"],
        "Image": ["jpg", "jpeg", "png"],
        "DOCX": ["docx", "doc"],
        "TXT": ["txt"],
    }

    uploaded_file = st.file_uploader(
        f"Upload an ID document ({', '.join(file_extensions[file_type])})",
        type=file_extensions[file_type],
    )

    use_vision = st.checkbox("Use vision analysis (if available)", value=True)

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Extract text from document
            extracted_text = process_file(uploaded_file, file_type, uploaded_file.name)

            # Perform vision analysis if enabled
            vision_analysis = None
            if use_vision:
                uploaded_file.seek(0)  # Reset file pointer
                st.info("Performing visual analysis...")
                vision_analysis = perform_vision_analysis(
                    uploaded_file, uploaded_file.name
                )

            # Display extracted text in expandable section
            with st.expander("Raw Extracted Text"):
                st.text(extracted_text)

            if vision_analysis:
                with st.expander("Vision Analysis Results"):
                    st.markdown(vision_analysis)

            # Extract structured information
            st.info("Extracting structured information...")
            extracted_info = extract_id_card_info(
                extracted_text, uploaded_file.name, vision_analysis
            )

            # Display results
            st.subheader("Extracted Information")
            if "error" in extracted_info:
                st.error(f"Error: {extracted_info['error']}")
                if "raw_response" in extracted_info:
                    with st.expander("Raw LLM Response"):
                        st.text(extracted_info["raw_response"])
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Document Details")
                    st.markdown(
                        f"**Document Type:** {extracted_info.get('document_type', 'N/A')}"
                    )
                    st.markdown(
                        f"**Document Number:** {extracted_info.get('document_number', 'N/A')}"
                    )
                    st.markdown(
                        f"**Issue Date:** {extracted_info.get('issue_date', 'N/A')}"
                    )
                    st.markdown(
                        f"**Expiry Date:** {extracted_info.get('expiry_date', 'N/A')}"
                    )
                    st.markdown(
                        f"**Issuing Authority:** {extracted_info.get('issuing_authority', 'N/A')}"
                    )

                with col2:
                    st.markdown("### Personal Information")
                    st.markdown(
                        f"**Full Name:** {extracted_info.get('full_name', 'N/A')}"
                    )
                    st.markdown(
                        f"**Nationality:** {extracted_info.get('nationality', 'N/A')}"
                    )
                    st.markdown(
                        f"**Birth Date:** {extracted_info.get('birth_date', 'N/A')}"
                    )
                    st.markdown(f"**Gender:** {extracted_info.get('gender', 'N/A')}")
                    st.markdown(
                        f"**Birth Place:** {extracted_info.get('birth_place', 'N/A')}"
                    )

                if extracted_info.get("additional_info"):
                    st.markdown("### Additional Information")
                    st.markdown(extracted_info["additional_info"])

                # Show JSON output
                with st.expander("JSON Output"):
                    st.json(extracted_info)

                # Provide download link for the JSON file
                json_filename = f"{uploaded_file.name.split('.')[0]}_extracted.json"
                json_path = OUTPUT_DIR / json_filename

                if json_path.exists():
                    with open(json_path, "rb") as file:
                        st.download_button(
                            label="Download JSON file",
                            data=file,
                            file_name=json_filename,
                            mime="application/json",
                        )


if __name__ == "__main__":
    main()

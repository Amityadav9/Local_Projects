from textwrap import dedent
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
import streamlit as st
from agno.models.ollama import Ollama

# Set up the Streamlit app
st.title("AI Travel Planner using Local LLM ✈️")
st.caption(
    "Plan your next adventure with AI Travel Planner by researching and planning a personalized itinerary on autopilot using local LLM"
)

# Get SerpAPI key from the user
serp_api_key = st.text_input(
    "Enter Serp API Key for Search functionality", type="password"
)

# Simple model selection with a hardcoded default
model_name = st.text_input(
    "Enter Ollama model name (e.g., llama3.2:3b)", value="llama3.2:3b"
)

if serp_api_key:
    researcher = Agent(
        name="Researcher",
        role="Searches for travel destinations, activities, and accommodations based on user preferences",
        model=Ollama(id=model_name),  # Changed 'model' to 'id'
        description=dedent(
            """\
        You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
        generate a list of search terms for finding relevant travel activities and accommodations.
        Then search the web for each term, analyze the results, and return the 10 most relevant results.
        """
        ),
        instructions=[
            "Given a travel destination and the number of days the user wants to travel for, first generate a list of 3 search terms related to that destination and the number of days.",
            "For each search term, `search_google` and analyze the results.",
            "From the results of all searches, return the 10 most relevant results to the user's preferences.",
            "Remember: the quality of the results is important.",
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )
    planner = Agent(
        name="Planner",
        role="Generates a draft itinerary based on user preferences and research results",
        model=Ollama(id=model_name),  # Changed 'model' to 'id'
        description=dedent(
            """\
        You are a senior travel planner. Given a travel destination, the number of days the user wants to travel for, and a list of research results,
        your goal is to generate a draft itinerary that meets the user's needs and preferences.
        """
        ),
        instructions=[
            "Given a travel destination, the number of days the user wants to travel for, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
            "Ensure the itinerary is well-structured, informative, and engaging.",
            "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
            "Remember: the quality of the itinerary is important.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
        ],
        add_datetime_to_instructions=True,
    )

    # Input fields for the user's destination and the number of days they want to travel for
    destination = st.text_input("Where do you want to go?")
    num_days = st.number_input(
        "How many days do you want to travel for?", min_value=1, max_value=30, value=7
    )

    if st.button("Generate Itinerary"):
        if not destination:
            st.error("Please enter a destination.")
        else:
            with st.spinner("Researching your destination..."):
                try:
                    # First get research results
                    research_results = researcher.run(
                        f"Research {destination} for a {num_days} day trip",
                        stream=False,
                    )

                    # Show research progress
                    st.write("✓ Research completed")

                    with st.spinner("Creating your personalized itinerary..."):
                        # Pass research results to planner
                        prompt = f"""
                        Destination: {destination}
                        Duration: {num_days} days
                        Research Results: {research_results.content}
                        
                        Please create a detailed itinerary based on this research.
                        """
                        response = planner.run(prompt, stream=False)
                        st.write(response.content)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info(
                        "Make sure Ollama is running and the specified model is installed."
                    )

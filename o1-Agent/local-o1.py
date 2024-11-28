# multi-agent system to build to build our own o1 agent/assistant
# Chain of thought Propmting

import ollama
import streamlit as st
import os
from datetime import datetime
import pathlib


# Initial CEO agent that creates the high-level plan
def CEO_AGENT(question, agent_responses=None):
    try:
        # If no agent responses, this is the initial plan
        if agent_responses is None:
            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "system",
                        "content": """You are O1, an advanced AI reasoning assistant 
                        focused on clearly and logically reasoning step by step analysis.
                        You excel at breaking down complex problems, thinking step by step,
                        and providing detailed explanations for your thought process. 
                        Always approach problems methodically and show your reasoning clearly.
                        Always answer in short.""",
                    },
                    {"role": "user", "content": question},
                ],
            )
        else:
            # If agent responses provided, this is the final review
            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {
                        "role": "system",
                        "content": """You are O1. Review your original plan and the implementations. 
                        Provide a final executive summary.""",
                    },
                    {
                        "role": "user",
                        "content": f"""Original Plan: {question}
                        
                        Agent Reports:
                        Step 1: {agent_responses[0]}
                        Step 2: {agent_responses[1]}
                        Step 3: {agent_responses[2]}
                        Step 4: {agent_responses[3]}
                        
                        Provide a concise summary of the complete implementation.""",
                    },
                ],
            )
        return response["message"]["content"]
    except Exception as e:
        st.error(f"Error communicating with AI model: {str(e)}")
        return None


# Factory function that creates specialized agents for each step
def create_step_agent(step_number):
    def agent(ceo_plan):
        return ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "system",
                    # Each agent only focuses on their numbered step from the plan
                    "content": f"""You are Agent {step_number}, reporting directly to the CEO. 
                Your only task is to take step {step_number} from the CEO's plan and explain 
                how you would implement it. Be specific and concise.Ignore all other steps.""",
                },
                {
                    "role": "user",
                    "content": f"CEO's plan:\n{ceo_plan}\n\nImplement step {step_number} only.",
                },
            ],
        )["message"]["content"]

    return agent


def save_summary(question, ceo_plan, implementations, final_review):
    try:
        # Create data directory if it doesn't exist
        data_dir = pathlib.Path("data")
        data_dir.mkdir(exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{timestamp}.txt"
        filepath = data_dir / filename

        # Save the summary with proper encoding
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("Original Question:\n")
            f.write(question + "\n\n")
            f.write("CEO's Initial Plan:\n")
            f.write(ceo_plan + "\n\n")
            f.write("Implementations:\n")
            for i, impl in enumerate(implementations, 1):
                f.write(f"Agent {i}'s Implementation:\n")
                f.write(impl + "\n\n")
            f.write("CEO's Final Review:\n")
            f.write(final_review)

        return filepath
    except Exception as e:
        st.error(f"Error saving summary: {str(e)}")
        return None


def main():
    st.title("AI Agent Implementation Plan")

    # Get user input with better UI
    question = st.text_area(
        "What would you like to ask the CEO Agent?",
        height=100,
        help="Enter your question here. Be specific for better results."
    )

    if st.button("Generate Plan") and question:
        if not question.strip():
            st.error("Please enter a question first.")
            return

        with st.spinner("Generating implementation plan..."):
            try:
                # Get initial plan from CEO
                ceo_plan = CEO_AGENT(question)
                if not ceo_plan:
                    return

                # Display CEO's initial plan in an expandable section
                with st.expander("CEO's Initial Plan", expanded=True):
                    st.write(ceo_plan)

                # Collect and display implementations from step agents
                implementations = []
                progress = st.progress(0)
                
                for i in range(1, 5):
                    agent = create_step_agent(i)
                    step_result = agent(ceo_plan)
                    if not step_result:
                        return
                    implementations.append(step_result)
                    progress.progress((i) * 0.2)

                    with st.expander(f"Agent {i}'s Implementation"):
                        st.write(step_result)

                # Get and display final review
                final_review = CEO_AGENT(ceo_plan, implementations)
                if not final_review:
                    return
                    
                with st.expander("CEO's Final Review", expanded=True):
                    st.write(final_review)

                # Save summary and show success message
                filepath = save_summary(question, ceo_plan, implementations, final_review)
                if filepath:
                    st.success(f"Summary saved to {filepath}")

                    # Add download button for the summary
                    try:
                        with open(filepath, "r", encoding='utf-8') as f:
                            st.download_button(
                                label="📥 Download Summary",
                                data=f.read(),
                                file_name=filepath.name,
                                mime="text/plain",
                            )
                    except Exception as e:
                        st.error(f"Error creating download button: {str(e)}")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()

import streamlit as st
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
from langchain_core.messages import HumanMessage  

# Load environment variables
load_dotenv()

def get_env_var(var: str):
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"{var} not found in environment variables. Make sure it is set in your .env file.")
    return value

# Load API keys
openai_api_key = get_env_var("OPENAI_API_COURSE_KEY")  
gemini_api_key = get_env_var("GEMINI_API_KEY")  
anthropic_api_key = get_env_var("ANTHROPIC_API_KEY")  
xai_api_key = get_env_var("XAI_API_KEY")

# Initialize LLMs
gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
gpt4o_mini_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
claude_chat = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, anthropic_api_key=anthropic_api_key)
genai.configure(api_key=gemini_api_key)  
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
grok_client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")

def query_grok(prompt: str):
    try:
        completion = grok_client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."},
                {"role": "user", "content": prompt}
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error querying Grok: {e}"

# Streamlit App Title
st.title("LLM & Agent Interaction App")

# Sidebar Configuration
st.sidebar.header("Configuration")
llm_provider = st.sidebar.selectbox("Choose LLM Provider", ["GPT-4o", "GPT-4o-mini", "Claude-3.5-Sonnet", "Gemini-2.0-Flash", "Grok-2-Latest"])
user_persona = st.sidebar.text_input("User Persona", "General User")
system_persona = st.sidebar.text_input("System Persona", "AI Assistant")
response_length = st.sidebar.radio("Response Length", ["Succinct", "Standard", "Thorough"], index=1)
temperature_setting = st.sidebar.radio("Conversation Type (Temperature)", ["Creative", "Balanced", "Precise"], index=1)
num_references = st.sidebar.slider("Number of Referenced Responses", 1, 10, 5)
follow_up_enabled = st.sidebar.checkbox("Enable Follow-up Queries")

user_input = st.text_area("Enter your prompt:")

if st.button("Submit Query"):
    if not user_input:
        st.warning("Please enter a prompt before submitting.")
    else:
        try:
            temperature_values = {"Creative": 0.8, "Balanced": 0.5, "Precise": 0.2}
            temperature = temperature_values[temperature_setting]
            
            if llm_provider == "GPT-4o":
                response = gpt4o_chat.invoke([HumanMessage(content=user_input)], temperature=temperature).content
            elif llm_provider == "GPT-4o-mini":
                response = gpt4o_mini_chat.invoke([HumanMessage(content=user_input)], temperature=temperature).content
            elif llm_provider == "Claude-3.5-Sonnet":
                response = claude_chat.invoke(user_input).content
            elif llm_provider == "Gemini-2.0-Flash":
                response = gemini_model.generate_content(user_input).text
            elif llm_provider == "Grok-2-Latest":
                response = query_grok(user_input)
            else:
                response = "Invalid model selection."
            
            st.success(f"{system_persona}: {response}")
            
            # Display referenced responses only if there are multiple references
            if num_references > 1:
                st.subheader("Referenced Responses")
                for i in range(min(num_references, 10)):
                    ref_response = f"Reference {i+1}: {response}"
                    st.write(ref_response)
            
            if follow_up_enabled:
                st.write("Suggested Follow-up Queries:")
                st.write("- Can you expand on that?")
                st.write("- What are some real-world applications?")
                st.write("- How does this compare to other approaches?")
        except Exception as e:
            st.error(f"Error: {e}")

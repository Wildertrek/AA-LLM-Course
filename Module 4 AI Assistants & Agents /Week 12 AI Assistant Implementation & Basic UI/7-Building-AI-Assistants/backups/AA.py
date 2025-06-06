import streamlit as st
import os
import re
import requests
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
tavily_api_key = get_env_var("TAVILY_API_KEY")

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

def query_tavily(search_query: str):
    """Perform a web search using Tavily API."""
    try:
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {tavily_api_key}", "Content-Type": "application/json"}
        payload = {"query": search_query}
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        return [f"Error querying Tavily: {e}"]

# Streamlit App Title
st.title("AI Assistant")

# Sidebar Configuration
st.sidebar.header("Configuration")
llm_provider = st.sidebar.selectbox("Choose LLM Provider", ["GPT-4o", "GPT-4o-mini", "Claude-3.5-Sonnet", "Gemini-2.0-Flash", "Grok-2-Latest"])
user_persona = st.sidebar.text_input("User Persona", "General User")
system_persona = st.sidebar.text_input("System Persona", "AI Assistant")
response_length = st.sidebar.radio("Response Length", ["Succinct", "Standard", "Thorough"], index=1)
temperature_setting = st.sidebar.radio("Conversation Type (Temperature)", ["Creative", "Balanced", "Precise"], index=1)
num_references = st.sidebar.slider("Number of Referenced Responses", 1, 10, 5)
follow_up_enabled = st.sidebar.checkbox("Enable Follow-up Queries")

# Initialize session state variables
if "process_query" not in st.session_state:
    st.session_state.process_query = False
if "web_search" not in st.session_state:
    st.session_state.web_search = False

# Query Interface at the Top
with st.container():
    st.subheader("Enter Your Query")
    user_input = st.text_area("Enter your prompt:", key="user_input")
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        if st.button("Submit Query"):
            st.session_state.process_query = True
    with col2:
        if st.button("Web Search with Tavily"):
            st.session_state.web_search = True

def process_query():
    if not st.session_state.user_input:
        st.warning("Please enter a prompt before submitting.")
    else:
        try:
            temperature_values = {"Creative": 0.8, "Balanced": 0.5, "Precise": 0.2}
            temperature = temperature_values[temperature_setting]
            
            if llm_provider == "GPT-4o":
                response = gpt4o_chat.invoke([HumanMessage(content=st.session_state.user_input)], temperature=temperature).content
            elif llm_provider == "GPT-4o-mini":
                response = gpt4o_mini_chat.invoke([HumanMessage(content=st.session_state.user_input)], temperature=temperature).content
            elif llm_provider == "Claude-3.5-Sonnet":
                response = claude_chat.invoke(st.session_state.user_input).content
            elif llm_provider == "Gemini-2.0-Flash":
                response = gemini_model.generate_content(st.session_state.user_input).text
            elif llm_provider == "Grok-2-Latest":
                response = query_grok(st.session_state.user_input)
            else:
                response = "Invalid model selection."
            
            st.subheader("Response")
            st.success(f"{system_persona}: {response}")
            
            # Store response in session state for copying
            st.session_state.response_text = f"**{system_persona}:**\n\n{response}"
            st.code(st.session_state.response_text, language='markdown')
            
            if st.button("Copy Response"):
                st.session_state.clipboard = st.session_state.response_text
                st.success("Response copied! You can paste it anywhere.")
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.process_query:
    process_query()
    st.session_state.process_query = False

if st.session_state.web_search:
    if not st.session_state.user_input:
        st.warning("Please enter a search query before submitting.")
    else:
        search_results = query_tavily(st.session_state.user_input)
        st.subheader("Tavily Search Results:")
        for idx, result in enumerate(search_results[:5]):  # Show top 5 results
            st.markdown(f"**{idx+1}. [{result['title']}]({result['url']})**")
            st.write(f"{result['content']}")
    st.session_state.web_search = False

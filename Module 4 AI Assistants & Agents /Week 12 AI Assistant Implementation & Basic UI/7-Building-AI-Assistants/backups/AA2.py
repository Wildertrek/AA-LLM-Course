import streamlit as st
import os
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

def generate_follow_up_queries(prompt: str, response: str):
    """Generate LLM-based follow-up queries based on the original query and response."""
    follow_up_prompt = f"Based on the following user query and response, suggest three relevant follow-up questions:\n\nUser Query: {prompt}\n\nResponse: {response}\n\nFollow-up Questions:"
    try:
        return gpt4o_chat.invoke([HumanMessage(content=follow_up_prompt)]).content.split("\n")
    except Exception as e:
        return [f"Error generating follow-up questions: {e}"]

# Function to process user query
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
            
            if follow_up_enabled:
                st.subheader("Follow-up Questions")
                follow_up_questions = generate_follow_up_queries(st.session_state.user_input, response)
                # Filter out empty strings and strip whitespace from each question
                filtered_questions = [q.strip() for q in follow_up_questions if q.strip()]
                for question in filtered_questions:
                    st.write(f"- {question}")
                
        except Exception as e:
            st.error(f"Error: {e}")

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
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Query Interface using a form with Ctrl+Enter support
st.subheader("Enter Your Query")

with st.form(key="query_form"):
    st.text_area("Enter your prompt:", key="user_input")
    
    # Create two columns for the buttons
    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        submit_button = st.form_submit_button("Submit Query")
    
    with col2:
        web_search_button = st.form_submit_button("Web Search with Tavily")
    
    if submit_button:
        st.session_state.process_query = True
    
    if web_search_button:
        if not st.session_state.user_input:
            st.warning("Please enter a search query before submitting.")
        else:
            search_results = query_tavily(st.session_state.user_input)
            st.session_state.search_results = search_results
            st.session_state.web_search = True

# Process the query if requested
if st.session_state.process_query:
    process_query()
    st.session_state.process_query = False

# Display search results if requested
if st.session_state.web_search and hasattr(st.session_state, 'search_results'):
    st.subheader("Tavily Search Results:")
    search_results = st.session_state.search_results
    if search_results:
        for idx, result in enumerate(search_results[:5]):  # Show top 5 results
            st.markdown(f"**{idx+1}. [{result['title']}]({result['url']})**")
            st.write(f"{result['content']}")
    else:
        st.write("No relevant search results found.")
    st.session_state.web_search = False
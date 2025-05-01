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

# =============================================================================
# Load Environment Variables
# =============================================================================
load_dotenv()

def get_env_var(var: str):
    """
    Retrieve an environment variable. If not found, raise a ValueError.
    """
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"{var} not found in environment variables. Make sure it is set in your .env file.")
    return value

# =============================================================================
# Retrieve API Keys from Environment Variables
# =============================================================================
openai_api_key = get_env_var("OPENAI_API_COURSE_KEY")  # Verify variable names
gemini_api_key = get_env_var("GEMINI_API_KEY")
anthropic_api_key = get_env_var("ANTHROPIC_API_KEY")
xai_api_key = get_env_var("XAI_API_KEY")
tavily_api_key = get_env_var("TAVILY_API_KEY")

# =============================================================================
# Initialize Other LLMs and API Clients
# =============================================================================
gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
gpt4o_mini_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
# Default Anthropic model; dynamic selection will override this if Anthropic Dynamic is chosen.
claude_chat = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, anthropic_api_key=anthropic_api_key)
genai.configure(api_key=gemini_api_key)  
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
grok_client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")

# =============================================================================
# Dynamic Model Listing Functions
# =============================================================================

@st.cache_data(show_spinner=False)
def get_available_openai_models(api_key: str):
    """
    Query the OpenAI API for available models and filter for those likely to support chat completions.
    """
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        if response.ok:
            models_data = response.json().get("data", [])
            # Filter for models that start with "gpt-" (adjust filtering if needed)
            chat_models = [m["id"] for m in models_data if m["id"].startswith("gpt-")]
            return chat_models
        else:
            st.error("Error retrieving OpenAI models: " + response.text)
            return []
    except Exception as e:
        st.error("Error retrieving OpenAI models: " + str(e))
        return []

@st.cache_data(show_spinner=False)
def get_available_anthropic_models(api_key: str):
    """
    Query the Anthropic API for available models and filter for those that support chat.
    Requires the "anthropic-version" header.
    """
    url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",  # Required header
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.ok:
            models_data = response.json().get("data", [])
            # Filter for models that start with "claude" (adjust as necessary)
            chat_models = [m["id"] for m in models_data if m["id"].startswith("claude")]
            return chat_models
        else:
            st.error("Error retrieving Anthropic models: " + response.text)
            return []
    except Exception as e:
        st.error("Error retrieving Anthropic models: " + str(e))
        return []

@st.cache_data(show_spinner=False)
def get_available_google_models(api_key: str):
    """
    Retrieve a list of available Google generative AI models.
    For API key authentication, the key is passed as a query parameter.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta2/models?key={api_key}"
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        if response.ok:
            data = response.json()
            # Assume response has a "models" key with list of model objects.
            models = data.get("models", [])
            # Extract model names (e.g., "models/gemini-2.0-flash")
            model_names = [model.get("name", "") for model in models if "name" in model]
            return model_names
        else:
            st.error("Error retrieving Google models: " + response.text)
            return []
    except Exception as e:
        st.error("Error retrieving Google models: " + str(e))
        return []

# =============================================================================
# Utility Functions for API Calls and Follow-Up Generation
# =============================================================================

def query_grok(prompt: str):
    """
    Query the Grok API using the provided prompt.
    """
    try:
        completion = grok_client.chat.completions.create(
            model="grok-2-latest",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error querying Grok: {e}"

def query_tavily(search_query: str, num_references: int):
    """
    Perform a web search using the Tavily API.
    """
    try:
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {tavily_api_key}", "Content-Type": "application/json"}
        payload = {"query": search_query, "num_results": num_references}
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        return [f"Error querying Tavily: {e}"]

def generate_follow_up_queries(prompt: str, response: str):
    """
    Generate three follow-up questions based on the user's prompt and LLM response.
    """
    follow_up_prompt = (
        "Based on the following user query and response, suggest three relevant follow-up questions:\n\n"
        f"User Query: {prompt}\n\nResponse: {response}\n\nFollow-up Questions:"
    )
    try:
        follow_up_text = gpt4o_chat.invoke([HumanMessage(content=follow_up_prompt)]).content
        questions = [q.strip() for q in follow_up_text.split("\n") if q.strip()]
        return questions
    except Exception as e:
        return [f"Error generating follow-up questions: {e}"]

# =============================================================================
# Streamlit Application Layout and Configuration Pane
# =============================================================================

st.title("AI Assistant")

# Sidebar: General configuration options.
st.sidebar.header("Configuration")
llm_provider = st.sidebar.selectbox(
    "Choose LLM Provider", 
    [
        "GPT-4o", 
        "GPT-4o-mini", 
        "Claude-3.5-Sonnet", 
        "Gemini-2.0-Flash", 
        "Grok-2-Latest", 
        "OpenAI Dynamic",
        "Anthropic Dynamic",
        "Google Dynamic"
    ]
)
user_persona = st.sidebar.text_input("User Persona", "General User")
system_persona = st.sidebar.text_input("System Persona", "AI Assistant")
response_length = st.sidebar.radio("Response Length", ["Succinct", "Standard", "Thorough"], index=1)
temperature_setting = st.sidebar.radio("Conversation Type (Temperature)", ["Creative", "Balanced", "Precise"], index=1)
num_references = st.sidebar.slider("Number of Referenced Responses", 1, 10, 5)
follow_up_enabled = st.sidebar.checkbox("Enable Follow-up Queries")

# Dynamic Dropdown for OpenAI models
if llm_provider == "OpenAI Dynamic":
    available_openai_models = get_available_openai_models(openai_api_key)
    if not available_openai_models:
        available_openai_models = ["gpt-4", "gpt-3.5-turbo"]
    selected_openai_model = st.sidebar.selectbox("Select an OpenAI Chat Model", available_openai_models)

# Dynamic Dropdown for Anthropic models
if llm_provider == "Anthropic Dynamic":
    available_anthropic_models = get_available_anthropic_models(anthropic_api_key)
    if not available_anthropic_models:
        available_anthropic_models = ["claude-3-5-sonnet-20241022", "claude-2", "claude-instant"]
    selected_anthropic_model = st.sidebar.selectbox("Select an Anthropic Chat Model", available_anthropic_models)

# Dynamic Dropdown for Google models
if llm_provider == "Google Dynamic":
    available_google_models = get_available_google_models(gemini_api_key)
    if not available_google_models:
        # Fallback static list; update as needed.
        available_google_models = ["models/gemini-2.0-flash"]
    selected_google_model = st.sidebar.selectbox("Select a Google Chat Model", available_google_models)

# =============================================================================
# Main Form for Query Input
# =============================================================================
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_area("Enter your prompt:")
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        submit_button = st.form_submit_button("Submit Query")
    with col2:
        web_search_button = st.form_submit_button("Web Search")

# =============================================================================
# Handling LLM Query Submission and Response Processing
# =============================================================================
if submit_button and user_input:
    st.session_state.user_input = user_input

    try:
        if llm_provider == "GPT-4o":
            response = gpt4o_chat.invoke([HumanMessage(content=user_input)]).content
        elif llm_provider == "GPT-4o-mini":
            response = gpt4o_mini_chat.invoke([HumanMessage(content=user_input)]).content
        elif llm_provider == "Claude-3.5-Sonnet":
            response = claude_chat.invoke([HumanMessage(content=user_input)]).content
        elif llm_provider == "Gemini-2.0-Flash":
            response_obj = gemini_model.generate(prompt=user_input)
            response = response_obj.text
        elif llm_provider == "Grok-2-Latest":
            response = query_grok(user_input)
        elif llm_provider == "OpenAI Dynamic":
            response = ChatOpenAI(
                model=selected_openai_model, 
                temperature=0, 
                openai_api_key=openai_api_key
            ).invoke([HumanMessage(content=user_input)]).content
        elif llm_provider == "Anthropic Dynamic":
            response = ChatAnthropic(
                model=selected_anthropic_model, 
                temperature=0, 
                anthropic_api_key=anthropic_api_key
            ).invoke([HumanMessage(content=user_input)]).content
        elif llm_provider == "Google Dynamic":
            google_model = genai.GenerativeModel(selected_google_model)
            response_obj = google_model.generate(prompt=user_input)
            response = response_obj.text
        else:
            response = "Unknown provider selected."
    except Exception as e:
        response = f"Error processing query: {e}"

    st.subheader("LLM Response:")
    st.write(response)

    if follow_up_enabled:
        follow_ups = generate_follow_up_queries(user_input, response)
        st.subheader("Follow-up Questions:")
        for idx, question in enumerate(follow_ups, start=1):
            st.markdown(f"**{idx}.** {question}")

# =============================================================================
# Handling Web Search via the Tavily API
# =============================================================================
if web_search_button:
    if not st.session_state.get("user_input"):
        st.warning("Please enter a search query before submitting.")
    else:
        search_results = query_tavily(st.session_state.user_input, num_references)
        st.subheader("Tavily Search Results:")
        if search_results:
            for idx, result in enumerate(search_results[:num_references]):
                st.markdown(f"**{idx+1}. [{result.get('title', 'No Title')}]({result.get('url', '#')})**")
                st.write(result.get("content", "No content available."))
        else:
            st.write("No relevant search results found.")

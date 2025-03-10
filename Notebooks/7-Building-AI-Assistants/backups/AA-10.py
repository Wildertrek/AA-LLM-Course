import streamlit as st
import os
import requests
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
from langchain_core.messages import HumanMessage

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# -----------------------------------------------------------------------------
# Load Environment Variables
# -----------------------------------------------------------------------------
load_dotenv()  # Load variables from a .env file into the environment

def get_env_var(var: str):
    """
    Retrieve an environment variable.
    Raise a ValueError if the variable is not found.
    """
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"{var} not found in environment variables. Ensure it is set in your .env file.")
    return value

# -----------------------------------------------------------------------------
# Retrieve API Keys from Environment Variables
# -----------------------------------------------------------------------------
openai_api_key = get_env_var("OPENAI_API_COURSE_KEY")
gemini_api_key = get_env_var("GEMINI_API_KEY")
anthropic_api_key = get_env_var("ANTHROPIC_API_KEY")
xai_api_key = get_env_var("XAI_API_KEY")
tavily_api_key = get_env_var("TAVILY_API_KEY")
# Vertex API-related keys and settings (add these to your .env)
vertex_api_key = get_env_var("VERTEX_API_KEY")
PROJECT_ID = get_env_var("PROJECT_ID")
VERTEX_LOCATION = get_env_var("VERTEX_LOCATION")  # e.g., "us-central1"

# -----------------------------------------------------------------------------
# Initialize LLMs and API Clients
# -----------------------------------------------------------------------------
gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
gpt4o_mini_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
claude_chat = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, anthropic_api_key=anthropic_api_key)
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
grok_client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")

# -----------------------------------------------------------------------------
# Dynamic Model Listing Functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_openai_models(api_key: str):
    """
    Query the OpenAI API for available models and filter for those likely to support chat completions.
    """
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            models_data = response.json().get("data", [])
            chat_models = [m["id"] for m in models_data if m["id"].startswith("gpt-")]
            return chat_models
        else:
            st.error("Error retrieving OpenAI models: " + response.text)
            logging.error("Error retrieving OpenAI models: %s", response.text)
            return []
    except Exception as e:
        st.error("Error retrieving OpenAI models: " + str(e))
        logging.exception("Exception retrieving OpenAI models:")
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
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            models_data = response.json().get("data", [])
            chat_models = [m["id"] for m in models_data if m["id"].startswith("claude")]
            return chat_models
        else:
            st.error("Error retrieving Anthropic models: " + response.text)
            logging.error("Error retrieving Anthropic models: %s", response.text)
            return []
    except Exception as e:
        st.error("Error retrieving Anthropic models: " + str(e))
        logging.exception("Exception retrieving Anthropic models:")
        return []

@st.cache_data(show_spinner=False)
def get_available_google_models(api_key: str):
    """
    Retrieve a list of available Google generative AI models.
    Implements a retry mechanism and timeout for improved robustness.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta2/models?key={api_key}"
    headers = {"Content-Type": "application/json"}
    max_retries = 3
    timeout = 10  # seconds
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.ok:
                data = response.json()
                models = data.get("models", [])
                model_names = [model.get("name", "") for model in models if "name" in model]
                return model_names
            else:
                logging.error("Attempt %d: Error retrieving Google models: %s", attempt, response.text)
        except Exception as e:
            logging.exception("Attempt %d: Exception retrieving Google models: %s", attempt, e)
        if attempt < max_retries:
            time.sleep(2)
    st.error("Error retrieving Google models after multiple attempts.")
    return []

@st.cache_data(show_spinner=False)
def get_available_vertex_models(project_id: str, location: str, vertex_api_key: str):
    """
    Retrieve a list of available Vertex AI models published by Google.
    """
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models"
    headers = {"Authorization": f"Bearer {vertex_api_key}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            data = response.json()
            models = data.get("models", [])
            # Return the model names (these are the model IDs used in API calls)
            model_names = [model.get("name", "") for model in models if "name" in model]
            return model_names
        else:
            st.error("Error retrieving Vertex models: " + response.text)
            logging.error("Error retrieving Vertex models: %s", response.text)
            return []
    except Exception as e:
        st.error("Error retrieving Vertex models: " + str(e))
        logging.exception("Exception retrieving Vertex models:")
        return []

def query_vertex_ai(prompt: str, project_id: str, location: str, model_id: str, vertex_api_key: str, temperature: float) -> str:
    """
    Query a Vertex AI model using the generateContent endpoint.
    """
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"
    headers = {
        "Authorization": f"Bearer {vertex_api_key}",
        "Content-Type": "application/json; charset=utf-8"
    }
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 256
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.ok:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            else:
                return "No response received."
        else:
            return f"Error: {response.text}"
    except Exception as e:
        logging.exception("Error querying Vertex AI:")
        return f"Error querying Vertex AI: {e}"

# -----------------------------------------------------------------------------
# Utility Functions for Other APIs
# -----------------------------------------------------------------------------
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
        logging.exception("Error querying Grok:")
        return f"Error querying Grok: {e}"

def query_tavily(search_query: str, num_references: int):
    """
    Perform a web search using the Tavily API.
    """
    try:
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {tavily_api_key}", "Content-Type": "application/json"}
        payload = {"query": search_query, "num_results": num_references}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        logging.exception("Error querying Tavily:")
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
        logging.exception("Error generating follow-up questions:")
        return [f"Error generating follow-up questions: {e}"]

# -----------------------------------------------------------------------------
# Streamlit Application Layout and Configuration Pane
# -----------------------------------------------------------------------------
st.title("AI Assistant")

# Sidebar configuration options.
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
        "Google Dynamic",
        "Vertex AI Dynamic"
    ]
)
user_persona = st.sidebar.text_input("User Persona", "General User")
system_persona = st.sidebar.text_input("System Persona", "AI Assistant")
response_length = st.sidebar.radio("Response Length", ["Succinct", "Standard", "Thorough"], index=1)
temperature_setting = st.sidebar.radio("Conversation Type (Temperature)", ["Creative", "Balanced", "Precise"], index=1)
num_references = st.sidebar.slider("Number of Referenced Responses", 1, 10, 5)
follow_up_enabled = st.sidebar.checkbox("Enable Follow-up Queries")

# -----------------------------------------------------------------------------
# Dynamic Dropdowns for Model Selections
# -----------------------------------------------------------------------------
# OpenAI Dynamic Model Dropdown
if llm_provider == "OpenAI Dynamic":
    available_openai_models = get_available_openai_models(openai_api_key)
    if not available_openai_models:
        available_openai_models = ["gpt-4", "gpt-3.5-turbo"]
    selected_openai_model = st.sidebar.selectbox("Select an OpenAI Chat Model", available_openai_models)

# Anthropic Dynamic Model Dropdown
if llm_provider == "Anthropic Dynamic":
    available_anthropic_models = get_available_anthropic_models(anthropic_api_key)
    if not available_anthropic_models:
        available_anthropic_models = ["claude-3-5-sonnet-20241022", "claude-2", "claude-instant"]
    selected_anthropic_model = st.sidebar.selectbox("Select an Anthropic Chat Model", available_anthropic_models)

# Google Dynamic Model Dropdown
if llm_provider == "Google Dynamic":
    available_google_models = get_available_google_models(gemini_api_key)
    if not available_google_models:
         available_google_models = ["models/gemini-2.0-flash"]
    selected_google_model = st.sidebar.selectbox("Select a Google Chat Model", available_google_models)

# Vertex AI Dynamic Model Dropdown
if llm_provider == "Vertex AI Dynamic":
    available_vertex_models = get_available_vertex_models(PROJECT_ID, VERTEX_LOCATION, vertex_api_key)
    if not available_vertex_models:
         available_vertex_models = ["gemini-2.0-flash-001"]  # Fallback model ID
    selected_vertex_model = st.sidebar.selectbox("Select a Vertex AI Model", available_vertex_models)

# -----------------------------------------------------------------------------
# Main Form for Query Input
# -----------------------------------------------------------------------------
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_area("Enter your prompt:")
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        submit_button = st.form_submit_button("Submit Query")
    with col2:
        web_search_button = st.form_submit_button("Web Search")

# -----------------------------------------------------------------------------
# Handling LLM Query Submission and Response Processing
# -----------------------------------------------------------------------------
if submit_button and user_input:
    st.session_state.user_input = user_input
    try:
        temperature_values = {"Creative": 0.8, "Balanced": 0.5, "Precise": 0.2}
        temperature = temperature_values.get(temperature_setting, 0.5)
        if llm_provider == "GPT-4o":
            response = gpt4o_chat.invoke([HumanMessage(content=user_input)], temperature=temperature).content
        elif llm_provider == "GPT-4o-mini":
            response = gpt4o_mini_chat.invoke([HumanMessage(content=user_input)], temperature=temperature).content
        elif llm_provider == "Claude-3.5-Sonnet":
            response = claude_chat.invoke([HumanMessage(content=user_input)], temperature=temperature).content
        elif llm_provider == "Gemini-2.0-Flash":
            response_obj = gemini_model.generate(prompt=user_input)
            response = response_obj.text
        elif llm_provider == "Grok-2-Latest":
            response = query_grok(user_input)
        elif llm_provider == "OpenAI Dynamic":
            response = ChatOpenAI(
                model=selected_openai_model,
                temperature=temperature,
                openai_api_key=openai_api_key
            ).invoke([HumanMessage(content=user_input)]).content
        elif llm_provider == "Anthropic Dynamic":
            response = ChatAnthropic(
                model=selected_anthropic_model,
                temperature=temperature,
                anthropic_api_key=anthropic_api_key
            ).invoke([HumanMessage(content=user_input)]).content
        elif llm_provider == "Google Dynamic":
            google_model = genai.GenerativeModel(selected_google_model)
            response_obj = google_model.generate(prompt=user_input)
            response = response_obj.text
        elif llm_provider == "Vertex AI Dynamic":
            response = query_vertex_ai(user_input, PROJECT_ID, VERTEX_LOCATION, selected_vertex_model, vertex_api_key, temperature)
        else:
            response = "Unknown provider selected."
    except Exception as e:
        logging.exception("Error processing query:")
        response = f"Error processing query: {e}"
    
    st.subheader("LLM Response:")
    st.write(response)
    
    if follow_up_enabled:
        follow_ups = generate_follow_up_queries(user_input, response)
        st.subheader("Follow-up Questions:")
        for idx, question in enumerate(follow_ups, start=1):
            st.markdown(f"**{idx}.** {question}")

# -----------------------------------------------------------------------------
# Handling Web Search via the Tavily API
# -----------------------------------------------------------------------------
if web_search_button:
    if not st.session_state.get("user_input"):
        st.warning("Please enter a search query before submitting.")
    else:
        search_results = query_tavily(st.session_state.user_input, num_references)
        st.subheader("Tavily Search Results:")
        if search_results:
            for idx, result in enumerate(search_results[:num_references]):
                title = result.get('title', 'No Title')
                url = result.get('url', '#')
                content = result.get("content", "No content available.")
                st.markdown(f"**{idx+1}. [{title}]({url})**")
                st.write(content)
        else:
            st.write("No relevant search results found.")

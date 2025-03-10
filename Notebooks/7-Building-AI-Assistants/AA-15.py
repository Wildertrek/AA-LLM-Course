import streamlit as st
import streamlit.components.v1 as components
import os
import requests
import time
import logging
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types  # Import types from the SDK
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
    """Retrieve an environment variable; raises error if not found."""
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"{var} not found in environment variables. Ensure it is set in your .env file.")
    return value

# -----------------------------------------------------------------------------
# Retrieve API Keys and Settings from Environment Variables
# -----------------------------------------------------------------------------
openai_api_key = get_env_var("OPENAI_API_COURSE_KEY")
gemini_api_key = get_env_var("GOOGLE_API_KEY")  # Gemini Developer API key from AI Studio
anthropic_api_key = get_env_var("ANTHROPIC_API_KEY")
xai_api_key = get_env_var("XAI_API_KEY")
tavily_api_key = get_env_var("TAVILY_API_KEY")

# -----------------------------------------------------------------------------
# Initialize LLMs and API Clients for other providers
# -----------------------------------------------------------------------------
gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
gpt4o_mini_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
claude_chat = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, anthropic_api_key=anthropic_api_key)
# Create a Gemini client using the google-genai SDK
client = genai.Client(api_key=gemini_api_key)
# Initialize a Grok client for XAI (example usage)
grok_client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")

# -----------------------------------------------------------------------------
# Dynamic Model Listing Functions for other providers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_openai_models(api_key: str):
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
            return []
    except Exception as e:
        st.error("Error retrieving OpenAI models: " + str(e))
        return []

@st.cache_data(show_spinner=False)
def get_available_anthropic_models(api_key: str):
    url = "https://api.anthropic.com/v1/models"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            models_data = response.json().get("data", [])
            chat_models = [m["id"] for m in models_data if m["id"].startswith("claude")]
            return chat_models
        else:
            st.error("Error retrieving Anthropic models: " + response.text)
            return []
    except Exception as e:
        st.error("Error retrieving Anthropic models: " + str(e))
        return []

@st.cache_data(show_spinner=False)
def get_available_google_models_via_sdk(api_key: str):
    """
    Uses the google-genai SDK to list available Gemini models.
    """
    try:
        client = genai.Client(api_key=api_key)
        models_iter = client.models.list()
        model_names = [model.name for model in models_iter]
        return model_names
    except Exception as e:
        st.error("Error listing Google models via SDK: " + str(e))
        return []

@st.cache_data(show_spinner=False)
def get_available_xai_models(api_key: str):
    """
    Retrieve available XAI models from the XAI API.
    """
    url = "https://api.x.ai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            models_data = response.json().get("data", [])
            model_ids = [m["id"] for m in models_data if "id" in m]
            return model_ids
        else:
            st.error("Error retrieving XAI models: " + response.text)
            return []
    except Exception as e:
        st.error("Error retrieving XAI models: " + str(e))
        return []

# -----------------------------------------------------------------------------
# Utility Functions for Other APIs
# -----------------------------------------------------------------------------
def query_grok(prompt: str, model: str = "grok-2-latest"):
    try:
        completion = grok_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error querying Grok: {e}"

def query_tavily(search_query: str, num_references: int):
    try:
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {tavily_api_key}", "Content-Type": "application/json"}
        payload = {"query": search_query, "num_results": num_references}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        return [f"Error querying Tavily: {e}"]

def generate_follow_up_queries(prompt: str, response: str):
    follow_up_prompt = (
        "Based on the following user query and response, suggest three relevant follow-up questions:\n\n"
        f"User Query: {prompt}\n\nResponse: {response}\n\nFollow-up Questions:"
    )
    try:
        follow_up_text = gpt4o_chat.invoke([HumanMessage(content=follow_up_prompt)]).content
        return [q.strip() for q in follow_up_text.split("\n") if q.strip()]
    except Exception as e:
        return [f"Error generating follow-up questions: {e}"]

# -----------------------------------------------------------------------------
# Custom Function: Copy Button for Output as an Icon
# -----------------------------------------------------------------------------
def render_copy_button(text):
    """Render a small copy icon button that copies `text` to the clipboard."""
    safe_text = json.dumps(text)  # Escapes the text for JavaScript
    html = f"""
    <script>
    function copyText() {{
        navigator.clipboard.writeText({safe_text}).then(function() {{
            console.log('Copying to clipboard was successful!');
            alert('Output copied to clipboard!');
        }}, function(err) {{
            console.error('Could not copy text: ', err);
            alert('Failed to copy text.');
        }});
    }}
    </script>
    <div style="text-align: right; margin-top: 10px;">
        <button style="background: transparent; border: none; cursor: pointer; font-size: 20px;"
            title="Copy Output"
            onclick="copyText();">
            📋
        </button>
    </div>
    """
    components.html(html, height=80)


# -----------------------------------------------------------------------------
# Unified Sidebar: LLM & File I/O / RAG Pipeline Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("Configuration Panel")

# LLM Provider Configuration
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
        "XAI Dynamic"
    ]
)
user_persona = st.sidebar.text_input("User Persona", "General User")
system_persona = st.sidebar.text_input("System Persona", "AI Assistant")
response_length = st.sidebar.radio("Response Length", ["Succinct", "Standard", "Thorough"], index=1)
temperature_setting = st.sidebar.radio("Conversation Type (Temperature)", ["Creative", "Balanced", "Precise"], index=1)
num_references = st.sidebar.slider("Number of Referenced Responses", 1, 10, 5)
follow_up_enabled = st.sidebar.checkbox("Enable Follow-up Queries")

# Dynamic Model Dropdowns for various providers
if llm_provider == "OpenAI Dynamic":
    available_openai_models = get_available_openai_models(openai_api_key)
    selected_openai_model = st.sidebar.selectbox("Select an OpenAI Model", available_openai_models or ["gpt-4", "gpt-3.5-turbo"])
if llm_provider == "Anthropic Dynamic":
    available_anthropic_models = get_available_anthropic_models(anthropic_api_key)
    selected_anthropic_model = st.sidebar.selectbox("Select an Anthropic Model", available_anthropic_models or ["claude-3-5-sonnet-20241022", "claude-2", "claude-instant"])
if llm_provider == "Google Dynamic":
    available_google_models = get_available_google_models_via_sdk(gemini_api_key)
    selected_google_model = st.sidebar.selectbox("Select a Google Model", available_google_models or ["gemini-2.0-flash-001"])
if llm_provider == "XAI Dynamic":
    available_xai_models = get_available_xai_models(xai_api_key)
    selected_xai_model = st.sidebar.selectbox("Select a XAI Model", available_xai_models or ["grok-2-latest"])

# RAG Pipeline File I/O Configuration
st.sidebar.markdown("### RAG Pipeline Configuration")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (images, PDFs, text files, etc.)",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png", "pdf", "txt"]
)
embedding_model = st.sidebar.selectbox(
    "Choose Embedding Model",
    ["text-embedding-004", "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"]
)
index_name = st.sidebar.text_input("Vector DB Index Name", value="my_vector_index")
if st.sidebar.button("Process Files"):
    st.sidebar.info("Processing uploaded files for RAG pipeline...")
    for uploaded_file in uploaded_files:
        st.sidebar.write(f"Processing file: {uploaded_file.name}")
        # Insert your file processing, embedding generation, and indexing logic here.
    st.sidebar.success("Files processed and indexed successfully!")

# -----------------------------------------------------------------------------
# Main Area: Query Input and LLM Response
# -----------------------------------------------------------------------------
st.title("AI Assistant")
# Preserve the last prompt by not clearing the form on submit.
with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_area("Enter your prompt:", value=st.session_state.get("user_input", ""))
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        submit_button = st.form_submit_button("Submit Query")
    with col2:
        web_search_button = st.form_submit_button("Web Search")

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
            response_obj = genai.GenerativeModel("gemini-2.0-flash").generate(prompt=user_input)
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
            client = genai.Client(api_key=gemini_api_key)
            resp = client.models.generate_content(
                model=selected_google_model,
                contents=user_input,
                config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=256)
            )
            response = resp.text
        elif llm_provider == "XAI Dynamic":
            response_obj = grok_client.chat.completions.create(
                model=selected_xai_model,
                messages=[{"role": "user", "content": user_input}]
            )
            response = response_obj.choices[0].message.content
        else:
            response = "Unknown provider selected."
    except Exception as e:
        response = f"Error processing query: {e}"
    
    st.subheader("LLM Response:")
    st.markdown(response)
    render_copy_button(response)
    
    if follow_up_enabled:
        follow_ups = generate_follow_up_queries(user_input, response)
        st.subheader("Follow-up Questions:")
        for idx, question in enumerate(follow_ups, start=1):
            st.markdown(f"**{idx}.** {question}")

if web_search_button:
    if not user_input:
        st.warning("Please enter a search query before submitting.")
    else:
        search_results = query_tavily(user_input, num_references)
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

import streamlit as st
import streamlit.components.v1 as components
import os
import requests
import logging
import json
from dotenv import load_dotenv

# google-genai imports
from google import genai
from google.genai import types

# OpenAI and Anthropic imports
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# Load Environment Variables from .env
# -----------------------------------------------------------------------------
load_dotenv()

def get_env_var(var: str):
    """Retrieve an environment variable; raises error if not found."""
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"{var} not found in environment variables. Ensure it is set in your .env file.")
    return value

# -----------------------------------------------------------------------------
# Retrieve API Keys
# -----------------------------------------------------------------------------
openai_api_key = get_env_var("OPENAI_API_COURSE_KEY")
gemini_api_key = get_env_var("GOOGLE_API_KEY")
anthropic_api_key = get_env_var("ANTHROPIC_API_KEY")
xai_api_key = get_env_var("XAI_API_KEY")
tavily_api_key = get_env_var("TAVILY_API_KEY")

# -----------------------------------------------------------------------------
# Model Token Limits (approximate; update as needed)
# -----------------------------------------------------------------------------
MODEL_TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4o": 8192,
    "gpt-4o-mini": 4096,
    "claude-2": 100000,
    "claude-3-5-sonnet-20241022": 9000,
    "gemini-2.0-flash": 8192,
    "grok-2-latest": 4096,
}

def clamp_max_tokens(selected_model: str, requested_max: int) -> int:
    """
    Clamps the requested max tokens to the model's limit minus overhead.
    """
    overhead = 512
    default_limit = 4096
    max_limit = MODEL_TOKEN_LIMITS.get(selected_model, default_limit)
    safe_limit = max_limit - overhead if (max_limit - overhead) > 0 else max_limit
    return min(requested_max, safe_limit)

# -----------------------------------------------------------------------------
# Initialize LLM Clients
# -----------------------------------------------------------------------------
gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
gpt4o_mini_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
claude_chat = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, anthropic_api_key=anthropic_api_key)
google_genai_client = genai.Client(api_key=gemini_api_key)
grok_client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")

# -----------------------------------------------------------------------------
# New Function: OpenAI Chat Completion with Debug Headers
# -----------------------------------------------------------------------------
def openai_chat_completion(messages, model, temperature, top_p, max_tokens, api_key):
    """
    Calls OpenAI's Chat Completions API with the given parameters.
    Returns the response JSON and the HTTP response headers.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Convert SystemMessage and HumanMessage objects to dictionaries
    messages_payload = [{"role": m.role, "content": m.content} for m in messages]
    
    payload = {
        "model": model,
        "messages": messages_payload,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.ok:
        return response.json(), response.headers
    else:
        st.error(f"OpenAI Chat Completion error: {response.status_code} {response.text}")
        return None, response.text

# -----------------------------------------------------------------------------
# Helper Function: Build System + User Messages
# -----------------------------------------------------------------------------
def build_system_user_messages(user_input: str, final_max_tokens: int, system_persona: str, user_persona: str):
    """
    Returns a list with a system message (with instructions) and a human message (the user's prompt).
    """
    system_text = (
        f"{system_persona}\n\n"
        f"You have up to ~{final_max_tokens} tokens to answer. Please do NOT abruptly cut off; "
        f"if you need more detail, summarize gracefully. User persona: {user_persona}\n"
    )
    return [SystemMessage(content=system_text), HumanMessage(content=user_input)]

# -----------------------------------------------------------------------------
# Functions to Dynamically List Models for Each Provider
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_openai_models(api_key: str):
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            models_data = response.json().get("data", [])
            return [m["id"] for m in models_data if m["id"].startswith("gpt-")]
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
            return [m["id"] for m in models_data if m["id"].startswith("claude")]
        else:
            st.error("Error retrieving Anthropic models: " + response.text)
            return []
    except Exception as e:
        st.error("Error retrieving Anthropic models: " + str(e))
        return []

@st.cache_data(show_spinner=False)
def get_available_google_models_via_sdk(api_key: str):
    try:
        client = genai.Client(api_key=api_key)
        return [model.name for model in client.models.list()]
    except Exception as e:
        st.error("Error listing Google models via SDK: " + str(e))
        return []

@st.cache_data(show_spinner=False)
def get_available_xai_models(api_key: str):
    url = "https://api.x.ai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            models_data = response.json().get("data", [])
            return [m["id"] for m in models_data if "id" in m]
        else:
            st.error("Error retrieving XAI models: " + response.text)
            return []
    except Exception as e:
        st.error("Error retrieving XAI models: " + str(e))
        return []

# -----------------------------------------------------------------------------
# Utility Functions for External Queries
# -----------------------------------------------------------------------------
def query_grok(prompt: str, model: str = "grok-2-latest", llm_params: dict = None):
    try:
        final_max_tokens = clamp_max_tokens(model, llm_params.get("max_tokens", 1024))
        system_instructions = f"System: You have up to ~{final_max_tokens} tokens. Do not abruptly cut off; summarize if needed.\n"
        combined_prompt = system_instructions + prompt
        completion = grok_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=llm_params.get("temperature", 0.5),
            max_tokens=final_max_tokens,
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

def generate_follow_up_queries(prompt: str, response: str, llm_params: dict):
    """
    Generates follow-up questions using GPT-4o.
    """
    follow_up_prompt = (
        "Based on the following user query and response, suggest three relevant follow-up questions:\n\n"
        f"User Query: {prompt}\n\nResponse: {response}\n\nFollow-up Questions:"
    )
    try:
        final_max_tokens = clamp_max_tokens("gpt-4o", llm_params.get("max_tokens", 512))
        system_text = (
            f"You have ~{final_max_tokens} tokens. Provide exactly three short follow-up questions. Avoid abrupt cutoffs."
        )
        messages = [SystemMessage(content=system_text), HumanMessage(content=follow_up_prompt)]
        follow_up_text = gpt4o_chat.invoke(
            input=messages,
            temperature=llm_params.get("temperature", 0.5),
            max_tokens=final_max_tokens
        ).content
        return [q.strip() for q in follow_up_text.split("\n") if q.strip()]
    except Exception as e:
        return [f"Error generating follow-up questions: {e}"]

def render_copy_button(text):
    """Renders a button to copy the text to the clipboard."""
    safe_text = json.dumps(text)
    html = f"""
    <script>
    function copyText() {{
        navigator.clipboard.writeText({safe_text}).then(function() {{
            console.log('Copy successful!');
            alert('Output copied to clipboard!');
        }}, function(err) {{
            console.error('Copy failed:', err);
            alert('Failed to copy text.');
        }});
    }}
    </script>
    <div style="text-align: right; margin-top: 10px;">
        <button style="background: transparent; border: none; cursor: pointer; font-size: 20px;" title="Copy Output" onclick="copyText();">📋</button>
    </div>
    """
    components.html(html, height=80)

def save_chat_completion_log(completion_details, debug_headers, filename="chat_completions_log.json"):
    """
    Saves the chat completion details along with debug headers to a JSON log file.
    """
    log_entry = {
        "completion": completion_details,
        "debug_headers": dict(debug_headers)
    }
    
    logs = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                logs = json.load(f)
        except Exception as e:
            print(f"Error reading existing log file: {e}")
    
    logs.append(log_entry)
    
    try:
        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)
        print("Chat completion log saved.")
    except Exception as e:
        print(f"Error saving chat completion log: {e}")

# -----------------------------------------------------------------------------
# Presets for Response Length
# -----------------------------------------------------------------------------
LENGTH_PRESETS = {
    "Succinct": {"temperature": 0.2, "max_tokens": 600, "top_p": 1.0, "top_k": 40},
    "Standard": {"temperature": 0.5, "max_tokens": 1024, "top_p": 1.0, "top_k": 40},
    "Thorough": {"temperature": 0.7, "max_tokens": 2048, "top_p": 0.95, "top_k": 50}
}

# -----------------------------------------------------------------------------
# Sidebar: Configuration Panel
# -----------------------------------------------------------------------------
st.sidebar.title("Configuration Panel")

llm_provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    ["GPT-4o", "GPT-4o-mini", "Claude-3.5-Sonnet", "Gemini-2.0-Flash", "Grok-2-Latest",
     "OpenAI Dynamic", "Anthropic Dynamic", "Google Dynamic", "XAI Dynamic"],
    help="Select the LLM provider."
)

user_persona = st.sidebar.text_input("User Persona", "General User", help="User's role/identity.")
system_persona = st.sidebar.text_input("System Persona", "AI Assistant", help="AI assistant's persona.")

response_length = st.sidebar.radio("Response Length", list(LENGTH_PRESETS.keys()), help="Select a preset controlling response detail.")
preset = LENGTH_PRESETS[response_length]

temperature_slider = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, preset["temperature"], 0.1)
max_tokens_slider = st.sidebar.slider("Max Output Tokens", 50, 4096, preset["max_tokens"], 50)
top_p_slider = st.sidebar.slider("Top-p (Nucleus Sampling)", 0.0, 1.0, preset["top_p"], 0.05)
top_k_slider = st.sidebar.slider("Top-k", 0, 200, preset["top_k"], 5)

llm_params = {"temperature": temperature_slider, "max_tokens": max_tokens_slider, "top_p": top_p_slider, "top_k": top_k_slider}

# Additional dynamic model dropdowns
if llm_provider == "OpenAI Dynamic":
    available_openai_models = get_available_openai_models(openai_api_key)
    selected_openai_model = st.sidebar.selectbox("Select an OpenAI Model", available_openai_models or ["gpt-4", "gpt-3.5-turbo"], help="OpenAI models.")
if llm_provider == "Anthropic Dynamic":
    available_anthropic_models = get_available_anthropic_models(anthropic_api_key)
    selected_anthropic_model = st.sidebar.selectbox("Select an Anthropic Model", available_anthropic_models or ["claude-2", "claude-instant"], help="Anthropic models.")
if llm_provider == "Google Dynamic":
    available_google_models = get_available_google_models_via_sdk(gemini_api_key)
    selected_google_model = st.sidebar.selectbox("Select a Google Model", available_google_models or ["gemini-2.0-flash-001"], help="Google Gemini models.")
if llm_provider == "XAI Dynamic":
    available_xai_models = get_available_xai_models(xai_api_key)
    selected_xai_model = st.sidebar.selectbox("Select an XAI Model", available_xai_models or ["grok-2-latest"], help="XAI models.")

# Follow-up Queries Checkbox
follow_up_enabled = st.sidebar.checkbox("Enable Follow-up Queries", help="Generate follow-up questions after the response.")

# Number of Referenced Responses (for web search / RAG)
num_references = st.sidebar.slider("Number of Referenced Responses", 1, 10, 5, help="How many search/RAG references to retrieve.")

# RAG Pipeline Configuration
st.sidebar.markdown("### RAG Pipeline Configuration")
uploaded_files = st.sidebar.file_uploader("Upload files (images, PDFs, text files, etc.)", accept_multiple_files=True, type=["jpg", "jpeg", "png", "pdf", "txt"], help="Files for RAG pipeline.")
embedding_model = st.sidebar.selectbox("Choose Embedding Model", ["text-embedding-004", "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"], help="Embedding model.")
index_name = st.sidebar.text_input("Vector DB Index Name", value="my_vector_index", help="Vector DB identifier.")

if st.sidebar.button("Process Files", help="Embed and index uploaded files."):
    st.sidebar.info("Processing files...")
    for uploaded_file in uploaded_files:
        st.sidebar.write(f"Processing file: {uploaded_file.name}")
        # Insert file processing logic here.
    st.sidebar.success("Files processed and indexed.")

# -----------------------------------------------------------------------------
# Main Query Section
# -----------------------------------------------------------------------------
st.title("AI Assistant with System Prompt & Token Clamping")

with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_area("Enter your prompt:", value=st.session_state.get("user_input", ""), help="Type your query here.")
    # Place the buttons side by side using columns
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.form_submit_button("Submit Query")
    with col2:
        web_search_button = st.form_submit_button("Web Search", help="Perform a web search for context.")

if submit_button and user_input:
    st.session_state.user_input = user_input
    try:
        response = "No response generated."
        
        if llm_provider in ["GPT-4o", "GPT-4o-mini", "Claude-3.5-Sonnet", "OpenAI Dynamic", "Anthropic Dynamic"]:
            if llm_provider == "GPT-4o":
                final_max = clamp_max_tokens("gpt-4o", llm_params["max_tokens"])
                messages = build_system_user_messages(user_input, final_max, system_persona, user_persona)
                response = gpt4o_chat.invoke(
                    input=messages,
                    temperature=llm_params["temperature"],
                    max_tokens=final_max
                ).content

            elif llm_provider == "GPT-4o-mini":
                final_max = clamp_max_tokens("gpt-4o-mini", llm_params["max_tokens"])
                messages = build_system_user_messages(user_input, final_max, system_persona, user_persona)
                response = gpt4o_mini_chat.invoke(
                    input=messages,
                    temperature=llm_params["temperature"],
                    max_tokens=final_max
                ).content

            elif llm_provider == "Claude-3.5-Sonnet":
                final_max = clamp_max_tokens("claude-3-5-sonnet-20241022", llm_params["max_tokens"])
                messages = build_system_user_messages(user_input, final_max, system_persona, user_persona)
                response = claude_chat.invoke(
                    input=messages,
                    temperature=llm_params["temperature"],
                    max_tokens=final_max
                ).content

            elif llm_provider == "OpenAI Dynamic":
                final_model = selected_openai_model
                final_max = clamp_max_tokens(final_model, llm_params["max_tokens"])
                messages = build_system_user_messages(user_input, final_max, system_persona, user_persona)
                # Call our custom OpenAI chat completion function
                openai_resp, debug_headers = openai_chat_completion(
                    messages, final_model, llm_params["temperature"], llm_params["top_p"], final_max, openai_api_key
                )
                if openai_resp:
                    response = openai_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                    st.markdown("**Debug/Rate Limiting Headers:**")
                    st.code(json.dumps({h: debug_headers.get(h) for h in debug_headers}, indent=4))
                    # Save the chat completion to a log file
                    save_chat_completion_log(openai_resp, debug_headers)
                else:
                    response = f"Error: {debug_headers}"

            elif llm_provider == "Anthropic Dynamic":
                final_model = selected_anthropic_model
                final_max = clamp_max_tokens(final_model, llm_params["max_tokens"])
                messages = build_system_user_messages(user_input, final_max, system_persona, user_persona)
                dynamic_anthropic = ChatAnthropic(
                    model=final_model,
                    temperature=llm_params["temperature"],
                    anthropic_api_key=anthropic_api_key
                )
                response = dynamic_anthropic.invoke(input=messages, max_tokens=final_max).content

        elif llm_provider in ["Gemini-2.0-Flash", "Google Dynamic"]:
            if llm_provider == "Gemini-2.0-Flash":
                final_max = clamp_max_tokens("gemini-2.0-flash", llm_params["max_tokens"])
                system_instructions = (
                    f"{system_persona}\n\n"
                    f"You have up to ~{final_max} tokens. Summarize if needed; do not abruptly cut off.\n"
                    f"User persona: {user_persona}\n"
                )
                combined_prompt = system_instructions + "\n\n" + user_input
                gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                response_obj = gemini_model.generate(
                    prompt=combined_prompt,
                    temperature=llm_params["temperature"],
                    max_output_tokens=final_max,
                    top_p=llm_params["top_p"],
                    top_k=int(llm_params["top_k"])
                )
                response = response_obj.text

            elif llm_provider == "Google Dynamic":
                final_model = selected_google_model
                final_max = clamp_max_tokens(final_model, llm_params["max_tokens"])
                system_instructions = (
                    f"{system_persona}\n\n"
                    f"You have up to ~{final_max} tokens. Summarize if needed; do not abruptly cut off.\n"
                    f"User persona: {user_persona}\n"
                )
                combined_prompt = system_instructions + "\n\n" + user_input
                google_config = types.GenerateContentConfig(
                    temperature=llm_params["temperature"],
                    max_output_tokens=final_max,
                    top_p=llm_params["top_p"],
                    top_k=int(llm_params["top_k"])
                )
                google_client = genai.Client(api_key=gemini_api_key)
                resp = google_client.models.generate_content(
                    model=final_model,
                    contents=combined_prompt,
                    config=google_config
                )
                response = resp.text

        elif llm_provider == "Grok-2-Latest":
            final_max = clamp_max_tokens("grok-2-latest", llm_params["max_tokens"])
            system_instructions = f"System: You have up to ~{final_max} tokens. Do not abruptly cut off; summarize if needed.\n"
            combined_prompt = system_instructions + user_input
            response = query_grok(combined_prompt, "grok-2-latest", llm_params=llm_params)
        else:
            response = "Unknown provider selected."

    except Exception as e:
        response = f"Error processing query: {e}"

    st.subheader("LLM Response:")
    st.markdown(response)
    render_copy_button(response)

    if follow_up_enabled:
        follow_ups = generate_follow_up_queries(user_input, response, llm_params)
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

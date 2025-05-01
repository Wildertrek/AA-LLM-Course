import streamlit as st
import streamlit.components.v1 as components
import os
import requests
import logging
import json
import time
from dotenv import load_dotenv

# google-genai imports
from google import genai
from google.genai import types

# OpenAI and Anthropic imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
from openai_model_selector import openai_model_selector_sidebar

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# Load Environment Variables from .env
# -----------------------------------------------------------------------------
load_dotenv()  # This loads your .env file so you can fetch API keys below

def get_env_var(var: str):
    """
    Retrieve an environment variable; raises error if not found.
    We call this repeatedly for different environment variables below.
    """
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
# Functions to Dynamically List Models for Each Provider
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_openai_models(api_key: str):
    """
    Retrieves the list of models from OpenAI’s endpoint.
    Filters only those starting with 'gpt-'.
    """
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
    """
    Calls Anthropic’s model listing endpoint to retrieve available Claude models.
    Caches the result.
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
            return [m["id"] for m in models_data if m["id"].startswith("claude")]
        else:
            st.error("Error retrieving Anthropic models: " + response.text)
            return []
    except Exception as e:
        st.error("Error retrieving Anthropic models: " + str(e))
        return []

@st.cache_data(show_spinner=False)
def get_available_google_models_via_sdk(api_key: str):
    """
    Uses the google.genai SDK to list available Google models (PaLM, Gemini, etc.).
    """
    try:
        client = genai.Client(api_key=api_key)
        return [model.name for model in client.models.list()]
    except Exception as e:
        st.error("Error listing Google models via SDK: " + str(e))
        return []

@st.cache_data(show_spinner=False)
def get_available_xai_models(api_key: str):
    """
    Hypothetical call to X.AI (or some other provider) to list available models.
    """
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
# New Function: OpenAI Chat Completion with Debug Headers
# -----------------------------------------------------------------------------
def openai_chat_completion(messages, model, temperature, top_p, max_tokens, api_key):
    """
    Wraps OpenAI's Chat Completions API in a custom function.
    Returns the response JSON plus HTTP headers for rate-limit debugging.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Convert messages to OpenAI’s schema
    messages_payload = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            # Fallback
            role = getattr(m, "role", "user")
        content = getattr(m, "content", "")
        messages_payload.append({"role": role, "content": content})
    
    payload = {
        "model": model,
        "messages": messages_payload,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
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
    Returns a list with a system message (instructions/persona) and a user message (prompt).
    """
    system_text = (
        f"{system_persona}\n\n"
        f"You have up to ~{final_max_tokens} tokens to answer. Please do NOT abruptly cut off; "
        f"if you need more detail, summarize gracefully. User persona: {user_persona}\n"
    )
    return [SystemMessage(content=system_text), HumanMessage(content=user_input)]

# -----------------------------------------------------------------------------
# Utility Functions for External Queries
# -----------------------------------------------------------------------------
def query_grok(prompt: str, model: str, llm_params: dict):
    """
    Placeholder for a custom external LLM or inference service.
    """
    try:
        final_max_tokens = llm_params.get("max_tokens", 1024)
        system_instructions = f"System: You have up to ~{final_max_tokens} tokens. Do not abruptly cut off; summarize if needed.\n"
        combined_prompt = system_instructions + prompt
        # Hypothetical usage of 'grok_client'
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
    """
    Example function to do a web search via an external API (Tavily).
    """
    try:
        url = "https://api.tavily.com/search"
        headers = {"Authorization": f"Bearer {tavily_api_key}", "Content-Type": "application/json"}
        payload = {"query": search_query, "num_results": num_references}
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        return [f"Error querying Tavily: {e}"]

def generate_follow_up_queries(prompt: str, response: str, llm_params: dict, selected_model: str, api_key: str):
    """
    Generates short follow-up questions using (by default) OpenAI’s Chat model.
    """
    follow_up_prompt = (
        "Based on the following user query and response, suggest three relevant follow-up questions:\n\n"
        f"User Query: {prompt}\n\nResponse: {response}\n\nFollow-up Questions:"
    )
    try:
        final_max_tokens = llm_params.get("max_tokens", 512)
        system_text = (
            f"You have ~{final_max_tokens} tokens. Provide exactly three short follow-up questions. "
            f"Avoid abrupt cutoffs."
        )
        messages = [SystemMessage(content=system_text), HumanMessage(content=follow_up_prompt)]
        openai_client = ChatOpenAI(model=selected_model, openai_api_key=api_key, temperature=llm_params.get("temperature", 0.5))
        follow_up_text = openai_client.invoke(
            input=messages,
            temperature=llm_params.get("temperature", 0.5),
            max_tokens=final_max_tokens
        ).content
        return [q.strip() for q in follow_up_text.split("\n") if q.strip()]
    except Exception as e:
        return [f"Error generating follow-up questions: {e}"]

def render_copy_button(text):
    """
    Adds a copy-to-clipboard button next to the LLM's output.
    """
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
        <button style="background: transparent; border: none; cursor: pointer; font-size: 20px;" 
                title="Copy Output" onclick="copyText();">📋</button>
    </div>
    """
    components.html(html, height=80)

def save_chat_completion_log(completion_details, debug_headers, filename="chat_completions_log.json"):
    """
    Appends chat completion details and headers to a JSON log file for debugging or auditing.
    """
    log_entry = {
        "completion": completion_details,
        "debug_headers": dict(debug_headers),
        "usage": completion_details.get("usage", {})
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
# We only define 'llm_provider' selectbox ONCE here, avoiding duplicate widget IDs.
st.sidebar.title("Configuration Panel")

llm_provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    ["OpenAI Dynamic", "Anthropic Dynamic", "Google Dynamic", "XAI Dynamic"],
    help="Select the LLM provider."
)

user_persona = st.sidebar.text_input("User Persona", "General User", help="User's role/identity.")
system_persona = st.sidebar.text_input("System Persona", "AI Assistant", help="AI assistant's persona.")

response_length = st.sidebar.radio("Response Length", list(LENGTH_PRESETS.keys()), help="Select a preset for response detail.")
preset = LENGTH_PRESETS[response_length]

temperature_slider = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, preset["temperature"], 0.1)
max_tokens_slider = st.sidebar.slider("Max Output Tokens", 50, 4096, preset["max_tokens"], 50)
top_p_slider = st.sidebar.slider("Top-p (Nucleus Sampling)", 0.0, 1.0, preset["top_p"], 0.05)
top_k_slider = st.sidebar.slider("Top-k", 0, 200, preset["top_k"], 5)

llm_params = {
    "temperature": temperature_slider,
    "max_tokens": max_tokens_slider,
    "top_p": top_p_slider,
    "top_k": top_k_slider
}

# Conditional dropdowns for each provider:
if llm_provider == "OpenAI Dynamic":
    available_openai_models = get_available_openai_models(openai_api_key)
    selected_openai_model = st.sidebar.selectbox(
        "Select an OpenAI Model",
        available_openai_models or ["gpt-4", "gpt-3.5-turbo"],
        help="OpenAI models."
    )
    # If the user wants advanced logic, or a dynamic approach, we can also call openai_model_selector_sidebar() here:
    # from openai_model_selector import openai_model_selector_sidebar
    # selected_openai_model = openai_model_selector_sidebar()

elif llm_provider == "Anthropic Dynamic":
    available_anthropic_models = get_available_anthropic_models(anthropic_api_key)
    selected_anthropic_model = st.sidebar.selectbox(
        "Select an Anthropic Model",
        available_anthropic_models or ["claude-2", "claude-instant"],
        help="Anthropic models."
    )

elif llm_provider == "Google Dynamic":
    available_google_models = get_available_google_models_via_sdk(gemini_api_key)
    selected_google_model = st.sidebar.selectbox(
        "Select a Google Model",
        available_google_models or ["gemini-2.0-flash-001"],
        help="Google Gemini models."
    )

elif llm_provider == "XAI Dynamic":
    available_xai_models = get_available_xai_models(xai_api_key)
    selected_xai_model = st.sidebar.selectbox(
        "Select an XAI Model",
        available_xai_models or ["grok-2-latest"],
        help="XAI models."
    )

follow_up_enabled = st.sidebar.checkbox("Enable Follow-up Queries", help="Generate follow-up questions after the response.")

# RAG Pipeline Configuration
st.sidebar.markdown("### RAG Pipeline Configuration")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (images, PDFs, text files, etc.)",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png", "pdf", "txt"],
    help="Files for RAG pipeline."
)
embedding_model = st.sidebar.selectbox(
    "Choose Embedding Model",
    ["text-embedding-004", "all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"],
    help="Embedding model."
)
index_name = st.sidebar.text_input("Vector DB Index Name", value="my_vector_index", help="Vector DB identifier.")

if st.sidebar.button("Process Files", help="Embed and index uploaded files."):
    st.sidebar.info("Processing files...")
    for uploaded_file in uploaded_files:
        st.sidebar.write(f"Processing file: {uploaded_file.name}")
        # Insert file-processing or embedding logic here.
    st.sidebar.success("Files processed and indexed.")

# -----------------------------------------------------------------------------
# Main Query Section with Timing
# -----------------------------------------------------------------------------
st.title("AI Assistant with Dynamic Models & Token Clamping")

with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_area("Enter your prompt:", value=st.session_state.get("user_input", ""), help="Type your query here.")
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.form_submit_button("Submit Query")
    with col2:
        web_search_button = st.form_submit_button("Web Search", help="Perform a web search for context.")

if submit_button and user_input:
    st.session_state.user_input = user_input
    start_time = time.time()
    response = "No response generated."
    
    try:
        if llm_provider == "OpenAI Dynamic":
            final_model = selected_openai_model
            final_max = max_tokens_slider
            messages = build_system_user_messages(user_input, final_max, system_persona, user_persona)
            openai_resp, debug_headers = openai_chat_completion(
                messages,
                final_model,
                llm_params["temperature"],
                llm_params["top_p"],
                final_max,
                openai_api_key
            )
            if openai_resp:
                response = openai_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                # Show debug headers and usage
                st.markdown("**Debug/Rate Limiting Headers:**")
                st.code(json.dumps({h: debug_headers.get(h) for h in debug_headers}, indent=4))

                usage = openai_resp.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "N/A")
                completion_tokens = usage.get("completion_tokens", "N/A")
                total_tokens = usage.get("total_tokens", "N/A")
                st.markdown(f"**Token Usage:** Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

                save_chat_completion_log(openai_resp, debug_headers)
            else:
                response = f"Error: {debug_headers}"

        elif llm_provider == "Anthropic Dynamic":
            final_model = selected_anthropic_model
            final_max = max_tokens_slider
            messages = build_system_user_messages(user_input, final_max, system_persona, user_persona)
            dynamic_anthropic = ChatAnthropic(
                model=final_model,
                temperature=llm_params["temperature"],
                anthropic_api_key=anthropic_api_key
            )
            response = dynamic_anthropic.invoke(input=messages, max_tokens=final_max).content

        elif llm_provider == "Google Dynamic":
            final_model = selected_google_model
            final_max = max_tokens_slider
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

        elif llm_provider == "XAI Dynamic":
            final_model = selected_xai_model
            final_max = max_tokens_slider
            system_instructions = f"System: You have up to ~{final_max} tokens. Do not abruptly cut off; summarize if needed.\n"
            combined_prompt = system_instructions + user_input
            response = query_grok(combined_prompt, final_model, llm_params=llm_params)
        
        else:
            response = "Unknown provider selected."

    except Exception as e:
        response = f"Error processing query: {e}"
    
    elapsed_time = time.time() - start_time
    st.subheader("LLM Response:")
    st.markdown(response)
    render_copy_button(response)
    
    st.markdown(f"**Total Query Time Taken: {elapsed_time:.2f} seconds**")
    
    if follow_up_enabled and (llm_provider == "OpenAI Dynamic"):
        # If generating follow-ups, we default to OpenAI logic
        follow_ups = generate_follow_up_queries(user_input, response, llm_params, selected_openai_model, openai_api_key)
        st.subheader("Follow-up Questions:")
        for idx, question in enumerate(follow_ups, start=1):
            st.markdown(f"**{idx}.** {question}")

if web_search_button:
    if not user_input:
        st.warning("Please enter a search query before submitting.")
    else:
        search_start = time.time()
        search_results = query_tavily(user_input, num_references=5)
        search_elapsed = time.time() - search_start
        
        st.subheader("Tavily Search Results:")
        st.markdown(f"**Web Search Time Taken: {search_elapsed:.2f} seconds**")
        
        if search_results:
            for idx, result in enumerate(search_results[:5]):
                title = result.get('title', 'No Title')
                url = result.get('url', '#')
                content = result.get("content", "No content available.")
                st.markdown(f"**{idx+1}. [{title}]({url})**")
                st.write(content)
        else:
            st.write("No relevant search results found.")

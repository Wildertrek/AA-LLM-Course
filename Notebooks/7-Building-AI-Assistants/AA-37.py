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

# LangChain / Anthropic
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------------------------------------------------------
# Model Configuration + Import from openai_model_selector
# -----------------------------------------------------------------------------
from openai_model_selector import (
    load_openai_model_metadata_as_dict,
    get_matched_openai_models
)

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# Load Environment Variables
# -----------------------------------------------------------------------------
load_dotenv()

def get_env_var(var: str):
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
# Chat Completion & Utility Functions
# -----------------------------------------------------------------------------
def openai_chat_completion(messages, model, temperature, top_p, max_tokens, api_key):
    """
    Manually map SystemMessage/HumanMessage to correct role to avoid attribute errors.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages_payload = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            role = getattr(m, "role", "assistant")

        content = getattr(m, "content", "")
        messages_payload.append({"role": role, "content": content})

    payload = {
        "model": model,
        "messages": messages_payload,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.ok:
        return resp.json(), resp.headers
    else:
        st.error(f"OpenAI Chat Completion error: {resp.status_code} {resp.text}")
        return None, resp.text


def build_system_user_messages(user_input: str, final_max_tokens: int, system_persona: str, user_persona: str):
    system_text = (
        f"{system_persona}\n\n"
        f"You have up to ~{final_max_tokens} tokens to answer. Please do NOT abruptly cut off; "
        f"if you need more detail, summarize gracefully. User persona: {user_persona}\n"
    )
    return [SystemMessage(content=system_text), HumanMessage(content=user_input)]


def save_chat_completion_log(completion_details, debug_headers, filename="chat_completions_log.json"):
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


def render_copy_button(text):
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


def generate_follow_up_queries(prompt: str, response: str, llm_params: dict, selected_model: str, api_key: str):
    """
    Example approach to create short follow-up questions via OpenAI.
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


# -----------------------------------------------------------------------------
# Additional Providers (Anthropic, Google, XAI)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_anthropic_models(api_key: str):
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
# Presets for Response Length
# -----------------------------------------------------------------------------
LENGTH_PRESETS = {
    "Succinct": {"temperature": 0.2, "max_tokens": 600, "top_p": 1.0, "top_k": 40},
    "Standard": {"temperature": 0.5, "max_tokens": 1024, "top_p": 1.0, "top_k": 40},
    "Thorough": {"temperature": 0.7, "max_tokens": 2048, "top_p": 0.95, "top_k": 50}
}

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
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
top_p_slider = st.sidebar.slider("Top-p (Nucleus Sampling)", 0.0, 1.0, preset["top_p"], 0.05)
top_k_slider = st.sidebar.slider("Top-k", 0, 200, preset["top_k"], 5)

follow_up_enabled = st.sidebar.checkbox("Enable Follow-up Queries", help="Generate follow-up questions after the response.")

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# If OpenAI Dynamic, show a dropdown of GPT Models
# Also figure out the model's max_tokens from metadata
# -----------------------------------------------------------------------------
model_max_token_limit = 4096  # fallback
chosen_model_label = None
if llm_provider == "OpenAI Dynamic":
    st.sidebar.markdown("### Main Chat Model")
    metadata_dict = load_openai_model_metadata_as_dict()
    matched_models = get_matched_openai_models(openai_api_key, metadata_dict)

    model_options = []
    for mm in matched_models:
        base_label = f" => {mm['display_name']}" if mm["base_id"] else ""
        model_options.append(f"{mm['dynamic_id']}{base_label}")

    chosen_model_label = st.sidebar.selectbox(
        "Select an OpenAI Model (Snapshot-based)",
        model_options or ["No GPT models found"],
        help="Snapshots like gpt-4o-audio-preview-2024-12-17 map to base metadata"
    )

    dynamic_id = chosen_model_label.split(" => ")[0]
    matched_obj = next((m for m in matched_models if m["dynamic_id"] == dynamic_id), None)
    st.sidebar.markdown("**Selected Main Chat Model Info**")
    if matched_obj:
        st.sidebar.markdown(f"- **Dynamic ID:** `{matched_obj['dynamic_id']}`")
        base_id = matched_obj["base_id"] if matched_obj["base_id"] else "(No base ID)"
        st.sidebar.markdown(f"- **Base ID:** `{base_id}`")
        st.sidebar.markdown(f"- **Display Name:** {matched_obj['display_name']}")
        st.sidebar.markdown(f"- **Summary:** {matched_obj['summary']}")
        st.sidebar.markdown(f"- **Description:** {matched_obj['description']}")
        st.sidebar.markdown(f"- **Context Window:** {matched_obj['context_window']}")
        st.sidebar.markdown(f"- **Max Output Tokens:** {matched_obj['max_output_tokens']}")

        if st.sidebar.checkbox("Show Full GPT Metadata"):
            st.sidebar.json(matched_obj["raw_metadata"])

        # If the matched model has a "max_output_tokens" field, use it
        if matched_obj["max_output_tokens"]:
            model_max_token_limit = matched_obj["max_output_tokens"]
    else:
        st.sidebar.write("No base metadata found for that GPT model.")
elif llm_provider == "Anthropic Dynamic":
    # If you want to parse the max tokens from an Anthropic model, do so here
    # For demonstration, just fallback
    model_max_token_limit = 100000
elif llm_provider == "Google Dynamic":
    # If you have a known limit from google, set it, otherwise fallback
    model_max_token_limit = 8192
elif llm_provider == "XAI Dynamic":
    model_max_token_limit = 4096

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# Next define the dynamic slider for "Max Output Tokens"
# We'll default the "value" to the preset's "max_tokens" or the model limit, whichever is smaller
# -----------------------------------------------------------------------------
default_slider_value = min(model_max_token_limit, preset["max_tokens"])
max_tokens_slider = st.sidebar.slider(
    "Max Output Tokens",
    50,  # min
    model_max_token_limit,  # max
    default_slider_value,   # initial
    50,                     # step
    help="Dynamically set based on chosen model's max_output_tokens if available."
)

# We'll build llm_params now that we know the final max
llm_params = {
    "temperature": temperature_slider,
    "max_tokens": max_tokens_slider,
    "top_p": top_p_slider,
    "top_k": top_k_slider
}

# -----------------------------------------------------------------------------
# RAG Pipeline with Embedding Model
# -----------------------------------------------------------------------------
st.sidebar.markdown("### RAG Pipeline Configuration")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (images, PDFs, text files, etc.)",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png", "pdf", "txt"],
    help="Files for RAG pipeline."
)
index_name = st.sidebar.text_input("Vector DB Index Name", value="my_vector_index", help="Vector DB identifier.")

# Load embedding models from metadata
embedding_metadata_dict = load_openai_model_metadata_as_dict()
embedding_candidates = []
for model_id, info in embedding_metadata_dict.items():
    endpoints = info.get("endpoints", {})
    if "Embeddings" in endpoints and endpoints["Embeddings"] != "Not supported":
        embedding_candidates.append(model_id)

embedding_options = []
for ec in embedding_candidates:
    disp = embedding_metadata_dict[ec].get("display_name", ec)
    embedding_options.append(f"{ec} => {disp}")

chosen_embedding_label = st.sidebar.selectbox(
    "Choose Embedding Model (OpenAI)",
    embedding_options or ["No known embedding models found"],
    help="Auto-load from your openai_models_metadata.json for embedding usage"
)

embedding_id = chosen_embedding_label.split(" => ")[0]
embedding_info = embedding_metadata_dict.get(embedding_id, {})

st.sidebar.markdown("**Selected Embedding Model Info**")
if embedding_info:
    st.sidebar.markdown(f"- **Embedding Model ID:** `{embedding_id}`")
    disp_name = embedding_info.get("display_name", "(No name)")
    st.sidebar.markdown(f"- **Display Name:** {disp_name}")
    sumry = embedding_info.get("summary", "")
    st.sidebar.markdown(f"- **Summary:** {sumry}")
    cw = embedding_info.get("context_window", "N/A")
    st.sidebar.markdown(f"- **Context Window:** {cw}")

    if st.sidebar.checkbox("Show Embedding Full Metadata"):
        st.sidebar.json(embedding_info)
else:
    st.sidebar.write("No metadata found for this embedding model.")

if st.sidebar.button("Process Files", help="Embed and index uploaded files."):
    st.sidebar.info("Processing files with embedding model: " + embedding_id)
    for uploaded_file in uploaded_files:
        st.sidebar.write(f"Processing file: {uploaded_file.name}")
        # Insert your actual embedding or indexing logic here
    st.sidebar.success("Files processed and indexed.")

# -----------------------------------------------------------------------------
# Main Query Section
# -----------------------------------------------------------------------------
st.title("AI Assistant (with dynamic Max Tokens)")

with st.form(key="query_form", clear_on_submit=False):
    user_input = st.text_area(
        "Enter your prompt:",
        value=st.session_state.get("user_input", ""),
        help="Type your query here."
    )
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
            # We'll re-use the GPT model chosen in "chosen_model_label"
            if chosen_model_label:
                dyn_id = chosen_model_label.split(" => ")[0]
            else:
                dyn_id = "gpt-3.5-turbo"

            # If we built matched_models earlier
            matched_obj = next((m for m in matched_models if m["dynamic_id"] == dyn_id), None)
            final_model = dyn_id if matched_obj else "gpt-3.5-turbo"

            final_max = llm_params["max_tokens"]
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
                # Token usage
                usage = openai_resp.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "N/A")
                completion_tokens = usage.get("completion_tokens", "N/A")
                total_tokens = usage.get("total_tokens", "N/A")
                st.markdown(f"**Token Usage:** Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

                save_chat_completion_log(openai_resp, debug_headers)
            else:
                response = f"Error: {debug_headers}"

        elif llm_provider == "Anthropic Dynamic":
            # Insert your Anthropic logic here
            pass

        elif llm_provider == "Google Dynamic":
            # Insert your Google logic here
            pass

        elif llm_provider == "XAI Dynamic":
            # Insert your XAI logic here
            pass
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
        fu_model = final_model
        follow_ups = generate_follow_up_queries(user_input, response, llm_params, fu_model, openai_api_key)
        st.subheader("Follow-up Questions:")
        for idx, question in enumerate(follow_ups, start=1):
            st.markdown(f"**{idx}.** {question}")

if web_search_button:
    if not user_input:
        st.warning("Please enter a search query before submitting.")
    else:
        search_start = time.time()
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

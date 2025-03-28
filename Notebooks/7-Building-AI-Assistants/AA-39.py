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
import anthropic  # Official Anthropic client
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------------------------------------------------------
# Import warnings and ignore warning message from the Anthropic Count Tokens function
# -----------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", message=".*The `json` method is deprecated.*")


# -----------------------------------------------------------------------------
# Import from openai_model_selector AND anthropic_model_selector
# -----------------------------------------------------------------------------
from openai_model_selector import (
    load_openai_model_metadata_as_dict,
    get_matched_openai_models
)
from anthropic_model_selector import (
    load_anthropic_model_metadata_as_dict,
    get_matched_anthropic_models
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
    Manually map SystemMessage/HumanMessage to correct role for OpenAI.
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
            role = getattr(m, "role", "assistant")  # e.g. "assistant"

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
    """
    Build a system message + user message list for both OpenAI & Anthropic usage.
    """
    system_text = (
        f"{system_persona}\n\n"
        f"You have up to ~{final_max_tokens} tokens to answer. "
        f"Please do NOT abruptly cut off; if you need more detail, summarize gracefully. "
        f"User persona: {user_persona}\n"
    )
    return [SystemMessage(content=system_text), HumanMessage(content=user_input)]

def save_chat_completion_log(completion_details, debug_headers, filename="chat_completions_log.json"):
    """
    Save chat completion info and headers to a JSON file for debugging / auditing.
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

def render_copy_button(text):
    """
    Renders a small button to copy text to clipboard using HTML/JS.
    """
    safe_text = json.dumps(text)
    html = f"""
    <script>
    function copyText() {{
        navigator.clipboard.writeText({safe_text}).then(function() {{
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
    Generate short follow-up questions using an OpenAI model.
    """
    follow_up_prompt = (
        "Based on the following user query and response, suggest three relevant follow-up questions:\n\n"
        f"User Query: {prompt}\n\nResponse: {response}\n\nFollow-up Questions:"
    )
    try:
        final_max_tokens = llm_params.get("max_tokens", 512)
        system_text = (
            f"You have ~{final_max_tokens} tokens. Provide exactly three short follow-up questions. "
            "Avoid abrupt cutoffs."
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
# Anthropic Token Counting Helper
# -----------------------------------------------------------------------------
def _extract_system_and_messages_for_anthropic(messages):
    """
    Anthropic doesn't allow a "system" role in the messages array.
    Instead, pass system=... at top level, and only user/assistant in 'messages'.

    We'll do that by scanning the input messages:
      - If it's SystemMessage, store that as system_str (concatenate if multiple)
      - If it's user or assistant, store in the array
    """
    system_str = None
    anthro_msgs = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            # If we get multiple system messages, we can combine them.
            if system_str is None:
                system_str = msg.content
            else:
                system_str += f"\n{msg.content}"
        elif isinstance(msg, HumanMessage):
            anthro_msgs.append({"role": "user", "content": msg.content})
        else:
            # e.g. AssistantMessage or custom role
            role = getattr(msg, "role", "assistant")
            content = getattr(msg, "content", "")
            if role == "system":
                if system_str is None:
                    system_str = content
                else:
                    system_str += f"\n{content}"
            elif role == "assistant":
                anthro_msgs.append({"role": "assistant", "content": content})
            else:
                # fallback
                anthro_msgs.append({"role": "assistant", "content": content})

    return system_str, anthro_msgs

def anthropic_count_tokens(model: str, messages, api_key: str):
    """
    Estimate prompt (input) tokens via Anthropic's 'count_tokens' endpoint.
    If system messages exist, pass them as system=..., and only user/assistant
    roles in the 'messages' param.
    """
    client = anthropic.Anthropic(api_key=api_key)
    try:
        system_str, anthro_msgs = _extract_system_and_messages_for_anthropic(messages)
        resp = client.messages.count_tokens(
            model=model,
            system=system_str,
            messages=anthro_msgs
        )
        
        # Some versions of the Anthropic client return a dict, others might return an object 
        # that .json() produces a string. So let's unify carefully:
        raw = resp.json()  # This might be a dict, or might be a string
        if isinstance(raw, dict):
            data = raw
        else:
            # If it's actually a string, we parse it ourselves
            data = json.loads(raw)

        # Now data should be a dict, e.g. {"input_tokens": 57}
        return data.get("input_tokens", None)
    except Exception as e:
        st.warning(f"Error counting Anthropic tokens: {e}")
        return None


# -----------------------------------------------------------------------------
# Additional Providers (Google, XAI)
# -----------------------------------------------------------------------------
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

# We'll define a default model_max_token_limit:
model_max_token_limit = 4096
chosen_model_label = None

# -----------------------------------------------------------------------------
# If OpenAI Dynamic
# -----------------------------------------------------------------------------
if llm_provider == "OpenAI Dynamic":
    st.sidebar.markdown("### Main Chat Model (OpenAI)")

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

        if matched_obj["max_output_tokens"]:
            model_max_token_limit = matched_obj["max_output_tokens"]

# -----------------------------------------------------------------------------
# If Anthropic Dynamic
# -----------------------------------------------------------------------------
elif llm_provider == "Anthropic Dynamic":
    st.sidebar.markdown("### Main Chat Model (Anthropic)")

    anthro_metadata = load_anthropic_model_metadata_as_dict()
    matched_anthropic = get_matched_anthropic_models(anthropic_api_key, anthro_metadata)

    model_options = []
    for mm in matched_anthropic:
        base_label = f" => {mm['display_name']}" if mm["base_id"] else ""
        model_options.append(f"{mm['dynamic_id']}{base_label}")

    chosen_model_label = st.sidebar.selectbox(
        "Select an Anthropic Model (Snapshot-based)",
        model_options or ["No Claude models found"],
        help="Snapshots like claude-3-7-sonnet-20250219 map to base metadata"
    )

    dynamic_id = chosen_model_label.split(" => ")[0]
    matched_obj = next((m for m in matched_anthropic if m["dynamic_id"] == dynamic_id), None)

    st.sidebar.markdown("**Selected Anthropic Model Info**")
    if matched_obj:
        st.sidebar.markdown(f"- **Dynamic ID:** `{matched_obj['dynamic_id']}`")
        base_id = matched_obj["base_id"] if matched_obj["base_id"] else "(No base ID)"
        st.sidebar.markdown(f"- **Base ID:** `{base_id}`")
        st.sidebar.markdown(f"- **Display Name:** {matched_obj['display_name']}")
        st.sidebar.markdown(f"- **Summary:** {matched_obj['summary']}")
        st.sidebar.markdown(f"- **Description:** {matched_obj['description']}")
        st.sidebar.markdown(f"- **Context Window:** {matched_obj['context_window']}")
        st.sidebar.markdown(f"- **Max Output Tokens:** {matched_obj['max_output_tokens']}")

        if st.sidebar.checkbox("Show Full Anthropic Metadata"):
            st.sidebar.json(matched_obj["raw_metadata"])

        if matched_obj["max_output_tokens"]:
            model_max_token_limit = matched_obj["max_output_tokens"]
    else:
        st.sidebar.write("No base metadata found for that Anthropic model.")

# -----------------------------------------------------------------------------
# If Google Dynamic
# -----------------------------------------------------------------------------
elif llm_provider == "Google Dynamic":
    st.sidebar.markdown("### Main Chat Model (Google)")
    available_google_models = get_available_google_models_via_sdk(gemini_api_key)
    selected_google_model = st.sidebar.selectbox(
        "Select a Google Model",
        available_google_models or ["gemini-2.0-flash-001"],
        help="Google Gemini models."
    )
    # For demonstration, set model_max_token_limit to a known or fallback
    model_max_token_limit = 8192

# -----------------------------------------------------------------------------
# If XAI Dynamic
# -----------------------------------------------------------------------------
elif llm_provider == "XAI Dynamic":
    st.sidebar.markdown("### Main Chat Model (XAI)")
    available_xai_models = get_available_xai_models(xai_api_key)
    selected_xai_model = st.sidebar.selectbox(
        "Select an XAI Model",
        available_xai_models or ["grok-2-latest"],
        help="XAI models."
    )
    model_max_token_limit = 4096

st.sidebar.markdown("---")

# Now define the dynamic slider for max output tokens
default_slider_value = min(model_max_token_limit, preset["max_tokens"])
max_tokens_slider = st.sidebar.slider(
    "Max Output Tokens",
    50,
    model_max_token_limit,
    default_slider_value,
    50,
    help="Dynamically set based on chosen model's max_output_tokens if available."
)

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

# We'll keep using OpenAI embeddings for now
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
    help="Loads from openai_models_metadata.json"
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
        # Insert your actual embedding or indexing logic
    st.sidebar.success("Files processed and indexed.")

# -----------------------------------------------------------------------------
# Main Query Section
# -----------------------------------------------------------------------------
st.title("AI Assistant with OpenAI / Anthropic & Prompt Token Counting")

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
        final_max = llm_params["max_tokens"]

        if llm_provider == "OpenAI Dynamic":
            # Use the chosen OpenAI model
            if chosen_model_label:
                dyn_id = chosen_model_label.split(" => ")[0]
            else:
                dyn_id = "gpt-3.5-turbo"

            matched_obj = next((m for m in matched_models if m["dynamic_id"] == dyn_id), None)
            final_model = dyn_id if matched_obj else "gpt-3.5-turbo"

            # Build messages
            messages = build_system_user_messages(user_input, final_max, system_persona, user_persona)

            # Call OpenAI
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
                usage = openai_resp.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "N/A")
                completion_tokens = usage.get("completion_tokens", "N/A")
                total_tokens = usage.get("total_tokens", "N/A")
                st.markdown(f"**OpenAI Token Usage:** Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                save_chat_completion_log(openai_resp, debug_headers)
            else:
                response = f"Error: {debug_headers}"

        elif llm_provider == "Anthropic Dynamic":
            # Use the chosen Anthropic model
            if chosen_model_label:
                dyn_id = chosen_model_label.split(" => ")[0]
            else:
                dyn_id = "claude-instant-1"

            matched_obj = next((m for m in matched_anthropic if m["dynamic_id"] == dyn_id), None)
            final_model = dyn_id if matched_obj else "claude-instant-1"

            # Build messages
            messages = build_system_user_messages(user_input, final_max, system_persona, user_persona)

            # 1) Count prompt tokens
            estimated_prompt_tokens = anthropic_count_tokens(final_model, messages, anthropic_api_key)
            if estimated_prompt_tokens is not None:
                st.markdown(f"**Anthropic Prompt Tokens (estimate):** {estimated_prompt_tokens}")

            # 2) Actually call the model
            dynamic_anthropic = ChatAnthropic(
                model=final_model,
                temperature=llm_params["temperature"],
                anthropic_api_key=anthropic_api_key
            )
            # For Anthropic, "max_tokens_to_sample" param is used in .invoke
            anthropic_resp = dynamic_anthropic.invoke(
                input=messages,
                max_tokens=final_max
            )
            response = anthropic_resp.content

        elif llm_provider == "Google Dynamic":
            # final_model = selected_google_model
            # ...
            pass

        elif llm_provider == "XAI Dynamic":
            # final_model = selected_xai_model
            # ...
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

    # Follow-up queries for OpenAI only
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

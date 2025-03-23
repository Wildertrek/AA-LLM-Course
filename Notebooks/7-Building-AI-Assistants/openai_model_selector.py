import streamlit as st
import json
import logging
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ----------------------------------------------------------------------------
# Load Environment Variables from .env
# ----------------------------------------------------------------------------
load_dotenv()

def get_env_var(var: str):
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"{var} not found in environment variables. Ensure it is set in your .env file.")
    return value

# ----------------------------------------------------------------------------
# Load OpenAI Model Metadata from JSON
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_openai_model_metadata():
    try:
        with open("openai_models_metadata.json", "r") as f:
            data = json.load(f)
            model_list = []
            for model_id, model_data in data.items():
                model_data["id"] = model_id  # add ID for UI and logic
                model_list.append(model_data)
        return model_list

    except Exception as e:
        st.error(f"Error loading OpenAI model metadata: {e}")
        return []

# ----------------------------------------------------------------------------
# Utility to Sort and Recommend Models
# ----------------------------------------------------------------------------
def rank_models(metadata, strategy="Price-weighted", price_weight=0.5, capability_weight=0.5):
    def score(model):
        price = model.get("price_score", 1.0)
        cap = model.get("capability_score", 1.0)
        if strategy == "Price-weighted":
            return price
        elif strategy == "Capability-weighted":
            return -cap
        elif strategy == "Combined":
            return price_weight * price - capability_weight * cap
        return price
    return sorted(metadata, key=score)

# ----------------------------------------------------------------------------
# Callable Function to Render Sidebar and Return Selected Model
# ----------------------------------------------------------------------------
def openai_model_selector_sidebar():
    model_metadata = load_openai_model_metadata()

    st.sidebar.title("Model Selection Settings")
    recommend_mode = st.sidebar.checkbox("Enable Auto-Recommend Mode")
    sort_strategy = st.sidebar.selectbox(
        "Ranking Strategy",
        ["Price-weighted", "Capability-weighted", "Combined"],
        help="Choose how to rank models."
    )

    if sort_strategy == "Combined":
        price_weight = st.sidebar.slider("Price Weight (if Combined)", 0.0, 1.0, 0.5, 0.05)
        capability_weight = st.sidebar.slider("Capability Weight (if Combined)", 0.0, 1.0, 0.5, 0.05)
    else:
        price_weight, capability_weight = 1.0, 1.0

    sorted_models = rank_models(model_metadata, sort_strategy, price_weight, capability_weight)

    selected_model = None
    if recommend_mode:
        top_model = sorted_models[0] if sorted_models else None
        if top_model:
            st.success(f"Auto-selected Model: {top_model['id']}")
            st.markdown(f"**Endpoint**: `{top_model.get('endpoint', 'unknown')}`")
            st.markdown(f"**Capabilities**: {', '.join(top_model.get('capabilities', []))}")
            st.markdown(f"**Price Score**: {top_model.get('price_score', 'N/A')} | Capability Score: {top_model.get('capability_score', 'N/A')}")
            selected_model = top_model['id']
    else:
        all_ids = [m["id"] for m in model_metadata]
        selected_id = st.sidebar.selectbox("Select a model", all_ids)
        selected_model = selected_id
        selected_model_data = next((m for m in model_metadata if m["id"] == selected_model), None)
        if selected_model_data:
            st.markdown(f"**Endpoint**: `{selected_model_data.get('endpoint', 'unknown')}`")
            st.markdown(f"**Capabilities**: {', '.join(selected_model_data.get('capabilities', []))}")
            st.markdown(f"**Price Score**: {selected_model_data.get('price_score', 'N/A')} | Capability Score: {selected_model_data.get('capability_score', 'N/A')}")

    # Tooltip Hover Info
    st.markdown("### Model Metadata Hover Tooltips")
    with st.expander("View All Models and Tooltips"):
        for model in model_metadata:
            tooltip = f"Endpoint: {model.get('endpoint')} | Price: {model.get('price_score')} | Capability: {model.get('capability_score')}"
            st.markdown(f"- **{model['id']}**")
            st.caption(tooltip)

    return selected_model

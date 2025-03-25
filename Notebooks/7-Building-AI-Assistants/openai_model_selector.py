# openai_model_selector.py

import streamlit as st
import json
import logging
import os
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
load_dotenv()


def get_env_var(var: str):
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"{var} not found in environment variables. Ensure it is set in your .env file.")
    return value


@st.cache_data(show_spinner=False)
def load_openai_model_metadata():
    """
    Loads openai_models_metadata.json, returning a LIST of model objects.
    (Used in older code or the original openai_model_selector_sidebar).
    """
    try:
        with open("openai_models_metadata.json", "r") as f:
            data = json.load(f)
            model_list = []
            for model_id, model_data in data.items():
                # Optionally store an 'id' if not present
                if "id" not in model_data:
                    model_data["id"] = model_id
                model_list.append(model_data)
        return model_list
    except Exception as e:
        st.error(f"Error loading OpenAI model metadata: {e}")
        return []


@st.cache_data(show_spinner=False)
def load_openai_model_metadata_as_dict():
    """
    Loads openai_models_metadata.json, returning a DICT keyed by base model ID.
    e.g. { "gpt-4o-audio-preview": {...}, "gpt-4o": {...}, etc. }
    """
    try:
        with open("openai_models_metadata.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading OpenAI model metadata as dictionary: {e}")
        return {}


def match_dynamic_model_to_metadata(model_id: str, metadata_dict: dict):
    """
    Given a dynamic ID like 'gpt-4o-audio-preview-2024-12-17',
    returns (base_key, base_info) from metadata_dict if:
      - model_id is a top-level key, or
      - model_id is in that base model's 'snapshots' array
    Otherwise returns (None, None).
    """
    # 1) Direct key match
    if model_id in metadata_dict:
        return model_id, metadata_dict[model_id]

    # 2) If not, check each base model's 'snapshots' array
    for base_key, base_info in metadata_dict.items():
        snapshots = base_info.get("snapshots", [])
        if model_id in snapshots:
            return base_key, base_info

    return None, None


@st.cache_data(show_spinner=False)
def get_matched_openai_models(api_key: str, metadata_dict: dict):
    """
    - Calls /v1/models to fetch 'gpt-*' dynamic IDs.
    - For each, we unify with the base model via match_dynamic_model_to_metadata.
    - Returns a list of dicts, each describing a dynamic model & its matched base info.
    """
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    matched_models = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.ok:
            all_data = resp.json().get("data", [])
            dynamic_ids = [m["id"] for m in all_data if m["id"].startswith("gpt-")]
            for dyn_id in dynamic_ids:
                base_key, base_info = match_dynamic_model_to_metadata(dyn_id, metadata_dict)
                if base_info:  # Found a match
                    matched_models.append({
                        "dynamic_id": dyn_id,
                        "base_id": base_key,  # date-less key from JSON
                        "display_name": base_info.get("display_name", dyn_id),
                        "summary": base_info.get("summary", ""),
                        "description": base_info.get("description", ""),
                        "context_window": base_info.get("context_window", None),
                        "max_output_tokens": base_info.get("max_output_tokens", None),
                        "raw_metadata": base_info
                    })
                else:
                    # No match found
                    matched_models.append({
                        "dynamic_id": dyn_id,
                        "base_id": None,
                        "display_name": dyn_id,
                        "summary": "",
                        "description": "",
                        "context_window": None,
                        "max_output_tokens": None,
                        "raw_metadata": {}
                    })
        else:
            st.error("Error retrieving OpenAI models: " + resp.text)
        return matched_models
    except Exception as e:
        st.error("Error retrieving OpenAI models: " + str(e))
        return matched_models


def rank_models(metadata, strategy="Price-weighted", price_weight=0.5, capability_weight=0.5):
    """
    Expects 'metadata' to be a list of model dicts, each with 'price_score'/'capability_score'.
    Returns a sorted list based on the chosen strategy.
    """
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


def openai_model_selector_sidebar():
    """
    Original approach if you want a simpler base-model selection
    rather than the snapshot-based approach.
    """
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
            tooltip = (
                f"Endpoint: {model.get('endpoint')} | "
                f"Price: {model.get('price_score')} | "
                f"Capability: {model.get('capability_score')}"
            )
            st.markdown(f"- **{model['id']}**")
            st.caption(tooltip)

    return selected_model

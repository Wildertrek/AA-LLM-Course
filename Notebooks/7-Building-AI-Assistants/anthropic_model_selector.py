# anthropic_model_selector.py

import streamlit as st
import json
import logging
import requests
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
load_dotenv()

def get_env_var(var: str):
    value = os.getenv(var)
    if value is None:
        raise ValueError(f"{var} not found in environment variables. Ensure it is set in your .env file.")
    return value

@st.cache_data(show_spinner=False)
def load_anthropic_model_metadata_as_dict():
    """
    Loads anthropic_models_metadata.json from the same directory.
    Returns a dictionary keyed by base model ID (like "claude-3-7-sonnet").
    """
    try:
        with open("anthropic_models_metadata.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading Anthropic model metadata: {e}")
        return {}

def match_dynamic_model_to_metadata(model_id: str, metadata_dict: dict):
    """
    If model_id is a top-level key, return (base_id, base_info).
    Else if model_id is in base_info['snapshots'], also return that.
    Otherwise (None, None).
    """
    # 1) Direct key match
    if model_id in metadata_dict:
        return model_id, metadata_dict[model_id]
    
    # 2) Check snapshots
    for base_key, base_info in metadata_dict.items():
        snapshots = base_info.get("snapshots", [])
        if model_id in snapshots:
            return base_key, base_info
    
    return None, None

@st.cache_data(show_spinner=False)
def get_matched_anthropic_models(api_key: str, metadata_dict: dict):
    """
    Calls Anthropic /v1/models to get dynamic model IDs,
    unifies them with base metadata if they start with 'claude'.
    Returns a list of matched model objects (similar to openai_model_selector).
    """
    url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    matched_models = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.ok:
            data = resp.json().get("data", [])
            # e.g. only keep model IDs that start with "claude"
            dyn_ids = [m["id"] for m in data if m["id"].startswith("claude")]
            
            for dyn_id in dyn_ids:
                base_key, base_info = match_dynamic_model_to_metadata(dyn_id, metadata_dict)
                if base_info:
                    matched_models.append({
                        "dynamic_id": dyn_id,
                        "base_id": base_key,
                        "display_name": base_info.get("display_name", dyn_id),
                        "summary": base_info.get("summary", ""),
                        "description": base_info.get("description", ""),
                        "context_window": base_info.get("context_window", None),
                        "max_output_tokens": base_info.get("max_output_tokens", None),
                        "raw_metadata": base_info
                    })
                else:
                    # Not found in JSON
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
            st.error(f"Error retrieving Anthropic models: {resp.status_code} {resp.text}")
    except Exception as e:
        st.error(f"Error retrieving Anthropic models: {e}")

    return matched_models

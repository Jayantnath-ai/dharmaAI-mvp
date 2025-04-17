import sys
import random
import os
import re

try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

try:
    import streamlit as st
    streamlit_available = True
except ImportError:
    streamlit_available = False

try:
    import pandas as pd
except ImportError:
    pd = None

if streamlit_available:
    st.set_page_config(page_title="DharmaAI MVP", layout="wide")

if openai_available:
    if not os.environ.get("OPENAI_API_KEY"):
        if streamlit_available:
            st.warning("⚠️ OpenAI API key not found. Krishna-GPT mode may not work.")
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    if streamlit_available:
        if "OPENAI_MODEL" not in st.session_state:
            st.session_state["OPENAI_MODEL"] = "gpt-3.5-turbo"
        available_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
        selected_model = st.sidebar.selectbox("🧠 Select OpenAI Model", available_models, index=available_models.index(st.session_state["OPENAI_MODEL"]))
        cost_per_1k = {
            "gpt-3.5-turbo": "$0.002 (input+output)",
            "gpt-4": "$0.05–$0.06 (est.)",
            "gpt-4o": "$0.02–$0.03 (est.)"
        }
        st.sidebar.caption(f"💰 Est. Cost per 1K tokens: {cost_per_1k.get(selected_model, 'Unknown')}")
        st.session_state["OPENAI_MODEL"] = selected_model

gemini_api_key = os.getenv("GEMINI_API_KEY")

# The rest of the original script continues here, keeping all logic intact.
# No Streamlit commands should be placed above `st.set_page_config`.

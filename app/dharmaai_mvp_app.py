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

# MVP App UI Starts Here
if streamlit_available:
    st.title("🪔 DharmaAI – Minimum Viable Conscience")

    if "ask_another" in st.session_state:
        st.session_state["user_input"] = ""
        if "Previous Questions" not in st.session_state:
            st.session_state["Previous Questions"] = []
        st.session_state["Previous Questions"].append("[Ask Another Clicked]")
        del st.session_state["ask_another"]
        st.experimental_rerun()

    mode = st.sidebar.radio("Select Mode", ["GitaBot", "Verse Matrix", "Usage Insights", "Scroll Viewer"])

    if mode == "GitaBot":
        st.header("🧠 GitaBot – Ask with Dharma")

        st.markdown("""
<style>
.ask-another-button {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background-color: #ffe082;
    padding: 0.75rem 1.5rem;
    border-radius: 2rem;
    color: black;
    text-align: center;
    font-weight: bold;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    cursor: pointer;
    transition: transform 0.2s ease, background-color 0.3s ease;
}
.ask-another-button:hover {
    background-color: #ffd54f;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

        st.markdown("<div class='ask-another-button'>", unsafe_allow_html=True)
        if st.button("🔄 Ask Another"):
            st.session_state["user_input"] = ""
            if "Previous Questions" not in st.session_state:
                st.session_state["Previous Questions"] = []
            st.session_state["Previous Questions"].append("[Ask Another Clicked]")
            st.markdown("""
<script>
    window.scrollTo({ top: 0, behavior: 'smooth' });
</script>
""", unsafe_allow_html=True)
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if "user_input" not in st.session_state:
            st.session_state["user_input"] = ""

        user_input = st.text_input("Ask a question or describe a dilemma:", value=st.session_state["user_input"], key="user_input")
        invocation_mode = st.selectbox(
            "Choose Invocation Mode",
            options=[
                "Krishna", "Krishna-GPT", "Krishna-Gemini",
                "Arjuna", "Vyasa", "Mirror", "Technical"
            ],
            index=0,
            format_func=lambda mode: {
                "Krishna": "🧠 Krishna – Classic dharma response",
                "Krishna-GPT": "🤖 Krishna-GPT – OpenAI-powered oracle",
                "Krishna-Gemini": "🌟 Krishna-Gemini – Gemini-powered reflection",
                "Arjuna": "😟 Arjuna – Human dilemma",
                "Vyasa": "📖 Vyasa – Epic narrator",
                "Mirror": "🪞 Mirror – See your own reflection",
                "Technical": "🔧 Technical – YAML debug mode"
            }.get(mode, mode)
        )

        submitted = st.button("🔍 Submit to GitaBot")
        clear = st.button("❌ Clear Question")

        if clear:
            st.session_state["user_input"] = ""
            st.experimental_rerun()

        if submitted and user_input:
            st.markdown(f"**Mode:** {invocation_mode}")
            st.markdown("---")
            response = generate_gita_response(invocation_mode, df_matrix, user_input)
            st.markdown(response)
            st.markdown("---")
            if "Usage Journal" in st.session_state and st.session_state["Usage Journal"]:
                latest = st.session_state["Usage Journal"][-1]
                st.caption(f"💬 Model: `{latest.get('model')}` | 🧮 Tokens: `{latest.get('tokens')}` | 💵 Cost: `${latest.get('cost_usd')}`")
                if 'verse_id' in latest:
                    st.caption(f"📘 Source: Verse {latest['verse_id']}")

        if "Previous Questions" not in st.session_state:
            st.session_state["Previous Questions"] = []
        if submitted and user_input:
            st.session_state["Previous Questions"].append(user_input)

        if st.session_state["Previous Questions"]:
            with st.expander("🕰 Recent Questions"):
                for i, q in enumerate(reversed(st.session_state["Previous Questions"][-5:])):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"- {q}")
                    with col2:
                        if st.button("🔁", key=f"reuse_{i}"):
                            st.session_state["user_input"] = q

    elif mode == "Verse Matrix":
        st.header("📜 Gita × DharmaAI Verse Matrix")
        st.write("(Matrix UI rendering placeholder)")

    elif mode == "Usage Insights":
        st.header("📊 Token & Cost Usage Journal")
        st.write("(Usage log rendering placeholder)")

    elif mode == "Scroll Viewer":
        st.header("📘 DharmaAI Scroll Library")
        st.write("(Scroll previews coming soon)")

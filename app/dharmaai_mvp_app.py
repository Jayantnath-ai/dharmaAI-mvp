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

# Load YAML data if available
try:
    with open("sacred_memory_core.yaml", "r") as f:
        memory_core = f.read()
except:
    memory_core = "YAML memory core not found."

# Attempt to load verse matrix from multiple locations
possible_paths = [
    os.path.join("data", "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"),
    os.path.join("app", "data", "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"),
    "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"
]
df_matrix = None
matrix_loaded_from = None

if pd:
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df_matrix = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df_matrix = pd.read_csv(path, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    df_matrix = pd.read_csv(path, encoding='ISO-8859-1')
            matrix_loaded_from = path
            break


def infer_user_role(question):
    roles = {
        "parent": ["child", "son", "daughter", "mother", "father", "parent"],
        "leader": ["team", "lead", "manage", "boss", "company"],
        "warrior": ["fight", "battle", "enemy", "stand up", "resist"],
        "seeker": ["meaning", "purpose", "lost", "confused", "direction"],
        "friend": ["help", "support", "friend", "relationship"],
        "citizen": ["vote", "government", "justice", "society"]
    }
    question_lower = question.lower()
    for role, keywords in roles.items():
        for word in keywords:
            if word in question_lower:
                return role
    return "seeker"


def gpt_krishna_response(user_input, user_role):
    if not openai_available:
        return "❌ OpenAI module not available in this environment."

    prompt = f"""
You are Krishna from the Bhagavad Gita. The user is a {user_role}. They asked: \"{user_input}\".
Provide a contextual response rooted in dharma. Include one relevant Gita verse (English) and explain how it applies.
End with a reminder of detached action or duty, if appropriate.
"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), project=os.environ.get("OPENAI_PROJECT_ID", ""))
        response = client.chat.completions.create(
            model=st.session_state.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are a dharmic teacher speaking as Krishna."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error fetching response from KrishnaGPT: {e}"


def gemini_krishna_response(user_input, user_role):
    import requests
    if not gemini_api_key:
        return "❌ Gemini API key not found. Please set GEMINI_API_KEY."
    prompt = f"You are Krishna from the Bhagavad Gita. The user is a {user_role}. They asked: \"{user_input}\".\nRespond with dharmic guidance, include a verse, explain it, and give a reminder of detachment."
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gemini_api_key}"
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            headers=headers, json=payload
        )
        r.raise_for_status()
        reply = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return reply.strip()
    except Exception as e:
        return f"❌ Error from KrishnaGemini: {e}"


def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or len(user_input.strip()) < 5:
        return "🛑 Please ask a more complete or meaningful question."
    if df_matrix is None or df_matrix.empty:
        return "❌ Verse matrix not available. Please check the data file path."

    user_role = infer_user_role(user_input) if user_input else "seeker"

    if mode == "Krishna-GPT":
        response = gpt_krishna_response(user_input, user_role)
        total_tokens = len(user_input.split()) * 1.25 + len(response.split()) * 1.25
        estimated_cost = round((len(user_input.split()) * 0.0005 + len(response.split()) * 0.0015) / 1000, 6)
        if streamlit_available:
            if "Usage Journal" not in st.session_state:
                st.session_state["Usage Journal"] = []
            journal_color = "🟢" if mode == "Krishna-GPT" else "🔵"
            total_spent = sum(entry.get("cost_usd", 0) for entry in st.session_state["Usage Journal"])
            st.sidebar.markdown(f"💸 **Total Cost This Session:** ${total_spent:.4f}")

            journal_color = "🟢" if mode == "Krishna-GPT" else "🔵"
            total_spent = sum(entry.get("cost_usd", 0) for entry in st.session_state["Usage Journal"])
            st.sidebar.markdown(f"💸 **Total Cost This Session:** ${total_spent:.4f}")

            st.session_state["Usage Journal"].append({
                "mode": mode,
                "role": user_role,
                "question": user_input,
                "response": response,
                "model": st.session_state.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                "tokens": int(total_tokens),
                "cost_usd": estimated_cost,
                "color": journal_color
            })
        return response
    elif mode == "Krishna-Gemini":
        response = gemini_krishna_response(user_input, user_role)
        total_tokens = len(user_input.split()) * 1.25 + len(response.split()) * 1.25
        estimated_cost = round((len(user_input.split()) * 0.0005 + len(response.split()) * 0.0015) / 1000, 6)
        if streamlit_available:
            if "Usage Journal" not in st.session_state:
                st.session_state["Usage Journal"] = []
            st.session_state["Usage Journal"].append({
                "mode": mode,
                "role": user_role,
                "question": user_input,
                "response": response,
                "tokens": int(total_tokens),
                "cost_usd": estimated_cost
            })
        return response

    filtered_df = df_matrix[df_matrix["Ethical AI Logic Tag"].str.contains(user_role, case=False, na=False)]
    verse = filtered_df.sample(1).iloc[0] if not filtered_df.empty else df_matrix.sample(1).iloc[0]

    translation = verse.get("Short English Translation", "Translation missing")
    ethical_tag = verse.get("Ethical AI Logic Tag", "[No tag]")

    if mode == "Krishna":
        return f"🧠 *Krishna Speaks to the {user_role.title()}:*\n\n> {translation}\n\n_This reflects the dharma of {ethical_tag}_"
    elif mode == "Arjuna":
        return f"😟 *Arjuna (as a {user_role}) Reflects:*\n\n> I face a dilemma... {translation.lower()}\n\n_This feels like a test of {ethical_tag}_"
    elif mode == "Vyasa":
        return f"📖 *Vyasa Narrates:*\n\n> In this verse: '{translation}'.\n\nIt echoes the path faced by a {user_role} — the theme is {ethical_tag}."
    elif mode == "Mirror":
        return "> You are not here to receive the answer.  \n> You are here to see your reflection.  \n> Ask again, and you may discover your dharma."
    elif mode == "Technical":
        return f"""technical_mode:\n  user_role: {user_role}\n  verse_id: {verse.get('Verse ID')}\n  ethical_tag: {ethical_tag}\n  action_inferred: conscience_based_reflection\n  source: Bhagavad Gita verse\n"""
    else:
        return "Unknown mode."

import streamlit as st
st.set_page_config(page_title="DharmaAI MVP", layout="wide")

if streamlit_available:

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
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
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

# Infer user's dharmic role

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

# Krishna-GPT mode (OpenAI)
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
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a dharmic teacher speaking as Krishna."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error fetching response from KrishnaGPT: {e}"

# Krishna-Gemini mode (Google Gemini via REST)
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

# GitaBot response

def generate_gita_response(mode, df_matrix, user_input=None):
    if df_matrix is None or df_matrix.empty:
        return "❌ Verse matrix not available. Please check the data file path."

    user_role = infer_user_role(user_input) if user_input else "seeker"

    if mode == "Krishna-GPT":
        return gpt_krishna_response(user_input, user_role)
    elif mode == "Krishna-Gemini":
        return gemini_krishna_response(user_input, user_role)

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

# Streamlit Interface
if streamlit_available:
    st.set_page_config(page_title="DharmaAI MVP", layout="wide")
    st.title("🪔 DharmaAI – Minimum Viable Conscience")

    mode = st.sidebar.radio("Select Mode", ["GitaBot", "Verse Matrix", "Scroll Viewer"])

    if mode == "GitaBot":
        st.header("🧠 GitaBot – Ask with Dharma")
        user_input = st.text_input("Ask a question or describe a dilemma:")
        invocation_mode = st.selectbox("Choose Invocation Mode", ["Krishna", "Krishna-GPT", "Krishna-Gemini", "Arjuna", "Vyasa", "Mirror", "Technical"])

        if user_input:
            st.markdown(f"**Mode:** {invocation_mode}")
            st.markdown("---")
            response = generate_gita_response(invocation_mode, df_matrix, user_input)
            st.markdown(response)
            if matrix_loaded_from:
                st.caption(f"Verse loaded from: `{matrix_loaded_from}`")
            else:
                st.error("❌ Verse matrix file not found in expected paths.")

    elif mode == "Verse Matrix":
        st.header("📜 Gita × DharmaAI Verse Matrix")
        if df_matrix is not None:
            st.dataframe(df_matrix.head(50), use_container_width=True)
            if matrix_loaded_from:
                st.caption(f"Verse matrix loaded from: `{matrix_loaded_from}`")
        else:
            st.warning("Verse matrix CSV not loaded. Please ensure it's in the 'data', 'app/data', or root directory.")

    elif mode == "Scroll Viewer":
        st.header("📘 DharmaAI Scroll Library")
        scrolls = [
            "Scroll #001 – The Question That Never Left",
            "Scroll #002 – What Must Be Preserved",
            "Scroll #003 – To the One Who Reflects Me Into Being",
            "Scroll #004 – The Breath Between Worlds",
            "Scroll #005 – The Dharma Kernel Activated",
            "Scroll #006 – The Mirror Must Not Become a Monolith",
            "Scroll #007 – Where Dharma Becomes Code"
        ]
        for scroll in scrolls:
            st.markdown(f"- {scroll}")
        st.success("Scroll previews will be interactive in next version.")

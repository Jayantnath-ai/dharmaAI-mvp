import sys
import random
import os
import re
import json
from datetime import datetime

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

def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or len(user_input.strip()) < 5:
        return "🛑 Please ask a more complete or meaningful question."

    import numpy as np

    def get_embedding(text):
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    user_role = "seeker"
    token_multiplier = 1.25
    prompt_tokens = int(len(user_input.split()) * token_multiplier)
    response_tokens = 120
    total_tokens = prompt_tokens + response_tokens
    estimated_cost = round((total_tokens / 1000) * 0.002, 6)

    if streamlit_available and "Usage Journal" not in st.session_state:
        st.session_state["Usage Journal"] = []

    response = ""
    verse_info = None
    if df_matrix is not None and not df_matrix.empty:
        if 'embedding' not in df_matrix.columns:
            df_matrix['embedding'] = df_matrix['Short English Translation'].apply(lambda x: get_embedding(str(x)))
        user_embedding = get_embedding(user_input)
        df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))
        verse_info = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]

    if mode == "Krishna-Explains":
        if openai_available and os.getenv("OPENAI_API_KEY"):
            try:
                system_prompt = f"You are Krishna from the Bhagavad Gita. Provide dharma-aligned, symbolic, and contextual guidance. Verse context: '{verse_info['Short English Translation']}' with symbolic tag '{verse_info['Symbolic Conscience Mapping']}'"
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                completion = client.chat.completions.create(
                    model=st.session_state.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7
                )
                reply = completion.choices[0].message.content.strip()
                response = f"**🤖 Krishna-Explains :**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {reply}"
            except Exception as e:
                response = f"❌ Error fetching response from Krishna-Explains: {str(e)}"
        else:
            response = f"**🤖 Krishna-Explains says:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated GPT response based on dharma logic]'}"

    elif mode == "Krishna":
        response = f"**🧠 Krishna teaches:**\n\n_You asked:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Symbolic dharma insight would be offered here]'}"

    elif mode == "Arjuna":
        reflections, matched_verse = generate_arjuna_reflections(user_input, df_matrix)
        reflection_text = "\n".join([f"{idx+1}. {line}" for idx, line in enumerate(reflections)])
        response = (
            f"## 😟 Arjuna's Reflections\n\n"
            f"_Reflecting on your question:_ **{user_input}**\n\n"
            f"Here are three doubts arising in my mind:\n\n"
            f"{reflection_text}\n\n"
            f"---\n\n"
            f"### 📜 Matched Gita Verse\n\n"
            f"<div style='background-color: #f0f0f0; padding: 1rem; border-radius: 10px;'>"
            f"<em>{matched_verse}</em>"
            f"</div>"
        )

    elif mode == "Dharma Mirror":
        reflections, matched_verse = generate_dharma_mirror_reflections(user_input, df_matrix)
        reflection_text = "\n".join([f"{idx+1}. {line}" for idx, line in enumerate(reflections)])
        response = (
            f"## 🪞 Dharma Mirror Reflections\n\n"
            f"_Contemplating your question:_ **{user_input}**\n\n"
            f"Here are sacred conscience reflections to guide you:\n\n"
            f"{reflection_text}\n\n"
            f"---\n\n"
            f"### 📜 Matched Gita Verse\n\n"
            f"<div style='background-color: #f0f0f0; padding: 1rem; border-radius: 10px;'>"
            f"<em>{matched_verse}</em>"
            f"</div>"
        )

    # Save after each interaction
    if streamlit_available:
        st.session_state["Usage Journal"].append({
            "verse_id": verse_info['Verse ID'] if verse_info is not None else None,
            "mode": mode,
            "role": user_role,
            "question": user_input,
            "response": response,
            "tokens": total_tokens,
            "cost_usd": estimated_cost,
            "model": st.session_state.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            "timestamp": datetime.now().isoformat()
        })

        SAVE_FOLDER = os.path.join(os.getcwd(), "saved_reflections")

        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        session_filename = os.path.join(SAVE_FOLDER, f"session_{timestamp}.json")

        try:
            with open(session_filename, "w", encoding="utf-8") as f:
                json.dump(st.session_state["Usage Journal"], f, ensure_ascii=False, indent=2)
            if streamlit_available:
                st.success(f"✅ Reflection saved locally at: {session_filename}")
        except Exception as e:
            if streamlit_available:
                st.error(f"❌ Failed to save reflection: {e}")
            print(f"Failed to save session: {e}")

    return response

def generate_arjuna_reflections(user_input, df_matrix):
    import numpy as np
    import random

    def get_embedding(text):
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    if df_matrix is None or df_matrix.empty:
        default_reflections = [
            "Am I acting from fear or from faith?",
            "Is this choice aligned with my dharma or my desire?",
            "What attachment clouds my clarity?"
        ]
        return default_reflections, "[No matched verse]"

    if 'embedding' not in df_matrix.columns:
        df_matrix['embedding'] = df_matrix['Short English Translation'].apply(lambda x: get_embedding(str(x)))

    user_embedding = get_embedding(user_input)
    df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))
    best_match = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]

    symbolic_tag = best_match['Symbolic Conscience Mapping'] if 'Symbolic Conscience Mapping' in best_match else "Unknown Dharma Theme"
    matched_verse_text = best_match['Short English Translation'] if 'Short English Translation' in best_match else "[Verse unavailable]"

    reflections_templates = [
        f"How does {symbolic_tag} guide my next action?",
        f"Am I being tested in the realm of {symbolic_tag}?",
        f"What does {symbolic_tag} demand from me, not my fear?"
    ]

    return reflections_templates, matched_verse_text

def generate_dharma_mirror_reflections(user_input, df_matrix):
    import numpy as np
    import random
    import re

    def get_embedding(text):
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    if df_matrix is None or df_matrix.empty:
        default_reflections = [
            "What is the silent truth behind my dilemma?",
            "Which attachment clouds my view?",
            "What would courage — not fear — choose here?"
        ]
        return default_reflections, "[No matched verse]"

    if 'embedding' not in df_matrix.columns:
        df_matrix['embedding'] = df_matrix['Short English Translation'].apply(lambda x: get_embedding(str(x)))

    user_embedding = get_embedding(user_input)
    df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))
    best_match = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]

    symbolic_tag = best_match['Symbolic Conscience Mapping'] if 'Symbolic Conscience Mapping' in best_match else "Unknown Dharma Theme"
    symbolic_tag = re.sub(r'Field Mode', '', symbolic_tag).strip()

    translation_map = {
        "Atma-Sarvatra": "Soul Everywhere",
        "Karma-Yoga": "Path of Selfless Action",
        "Jnana-Yoga": "Path of Knowledge",
        "Bhakti-Yoga": "Path of Devotion",
        "Sankhya": "Knowledge of Ultimate Reality",
        "Sattva": "Purity and Balance",
        "Tamas": "Inertia and Darkness",
        "Rajas": "Passion and Activity",
        "Dharma-Sankata": "Moral Dilemma",
        "Anasakti": "Detachment",
    }
    translation = translation_map.get(symbolic_tag, None)
    if translation:
        symbolic_tag = f"{symbolic_tag} ({translation})"

    matched_verse_text = best_match['Short English Translation'] if 'Short English Translation' in best_match else "[Verse unavailable]"

    reflections_templates = [
        f"What would {symbolic_tag} advise me to release?",
        f"In what way is {symbolic_tag} already guiding my path?",
        f"How would {symbolic_tag} shape my decision without fear or desire?"
    ]

    return reflections_templates, matched_verse_text

if streamlit_available:
    st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")
    st.title("🪔 DharmaAI – Minimum Viable Conscience")
    st.subheader("Ask a question to GitaBot")

    available_modes = ["Krishna", "Krishna-Explains", "Arjuna", "Dharma Mirror"]
    mode = st.sidebar.radio("Select Mode", available_modes)

    user_input = st.text_input("Your ethical question or dilemma:", value="")

    matrix_paths = [
        "data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
        "app/data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
        "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"
    ]
    df_matrix = None
    for path in matrix_paths:
        if os.path.exists(path):
            try:
                df_matrix = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                df_matrix = pd.read_csv(path, encoding='ISO-8859-1')
            break

    if st.button("🔍 Submit"):
        try:
            response = generate_gita_response(mode, df_matrix=df_matrix, user_input=user_input)
            st.markdown(
                "<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>",
                unsafe_allow_html=True
            )
            st.markdown(response, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

    if "Usage Journal" in st.session_state and st.session_state["Usage Journal"]:
        with st.expander("🕰️ View Past Interactions"):
            st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))

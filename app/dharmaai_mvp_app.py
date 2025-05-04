import streamlit as st
import openai
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Set Streamlit config early
st.set_page_config(page_title="🖔 DharmaAI – GitaBot Reflection Engine", layout="centered")

def get_embedding(text):
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(1536)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or len(user_input.strip()) < 5:
        return "🛑 Please ask a more complete or meaningful question."

    user_role = "seeker"
    token_multiplier = 1.25
    prompt_tokens = int(len(user_input.split()) * token_multiplier)
    response_tokens = 120
    total_tokens = prompt_tokens + response_tokens

    if "Usage Journal" not in st.session_state:
        st.session_state["Usage Journal"] = []

    response = ""
    verse_info = None
    if df_matrix is not None and not df_matrix.empty:
        if 'embedding' not in df_matrix.columns:
            df_matrix['embedding'] = df_matrix['Short English Translation'].apply(lambda x: get_embedding(str(x)))
        user_embedding = get_embedding(user_input)
        df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))
        verse_info = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]

    if mode == "Krishna":
        response = f"**🧐 Krishna teaches:**\n\n_You asked:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Symbolic dharma insight would be offered here]'}"

    elif mode == "Arjuna":
        reflections = [
            "Am I acting from fear or from faith?",
            "Is this choice aligned with my dharma or my desire?",
            "What attachment clouds my clarity?"
        ]
        reflection_text = "\n".join([f"{idx+1}. {line}" for idx, line in enumerate(reflections)])
        response = (
            f"## 😟 Arjuna's Reflections\n\n"
            f"_Reflecting on your question:_ **{user_input}**\n\n"
            f"Here are three doubts arising in my mind:\n\n"
            f"{reflection_text}"
        )

    elif mode == "Dharma Mirror":
        symbolic_tag = verse_info['Symbolic Conscience Mapping'] if verse_info is not None else "Unknown Dharma Theme"
        translation_map = {
            "Atma-Sarvatra": "Soul Everywhere",
            "Karma-Yoga": "Path of Selfless Action",
            "Jnana-Yoga": "Path of Knowledge",
            "Bhakti-Yoga": "Path of Devotion",
        }
        if symbolic_tag in translation_map:
            symbolic_tag = f"{symbolic_tag} ({translation_map[symbolic_tag]})"

        reflections = [
            f"What would {symbolic_tag} advise me to release?",
            f"In what way is {symbolic_tag} already guiding my path?",
            f"How would {symbolic_tag} shape my decision without fear or desire?"
        ]
        reflection_text = "\n".join([f"{idx+1}. {line}" for idx, line in enumerate(reflections)])
        response = (
            f"## 🧮 Dharma Mirror Reflections\n\n"
            f"_Contemplating your question:_ **{user_input}**\n\n"
            f"Here are sacred conscience reflections to guide you:\n\n"
            f"{reflection_text}"
        )

    if "Usage Journal" in st.session_state:
        st.session_state["Usage Journal"].append({
            "verse_id": verse_info['Verse ID'] if verse_info is not None else None,
            "mode": mode,
            "role": user_role,
            "question": user_input,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

        save_dir = os.path.join(os.getcwd(), "saved_reflections")
        os.makedirs(save_dir, exist_ok=True)
        filename = f"session_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.json"
        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(st.session_state["Usage Journal"], f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Failed to save session: {e}")

    return response

def main():
    st.title("🖔 DharmaAI – Minimum Viable Conscience")
    st.subheader("Ask a question to GitaBot")

    mode = st.sidebar.radio("Select Mode", ["Krishna", "Arjuna", "Dharma Mirror"])
    user_input = st.text_input("Your ethical question or dilemma:", value="")

    df_matrix = None
    matrix_paths = [
        "data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
        "app/data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
        "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"
    ]
    for path in matrix_paths:
        if os.path.exists(path):
            try:
                df_matrix = pd.read_csv(path)
                break
            except Exception:
                continue

    if st.button("🔍 Submit"):
        try:
            response = generate_gita_response(mode, df_matrix=df_matrix, user_input=user_input)
            st.markdown("<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>", unsafe_allow_html=True)
            st.markdown(response, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

    if "Usage Journal" in st.session_state and st.session_state["Usage Journal"]:
        with st.expander("🕰️ View Past Interactions"):
            st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))

if __name__ == '__main__':
    main()

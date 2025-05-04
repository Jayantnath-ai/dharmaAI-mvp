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

# 🔵 MAIN GITA RESPONSE GENERATOR
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

    # Save silently
    if streamlit_available:
        st.session_state["Usage Journal"].append({
            "verse_id": verse_info['Verse ID'] if verse_info is not None else None,
            "mode": mode,
            "role": user_role,
            "question": user_input,
            "response": response,
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
        except Exception as e:
            if streamlit_available:
                st.error(f"❌ Failed to save reflection: {e}")
            print(f"Failed to save session: {e}")

    return response

# 🔵 SUPPORT FUNCTIONS
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

def simulate_karmic_entanglement(dilemma):
    import random
    karmic_paths = {
        "Reveal Truth": {
            "tags": ["Satya (Truth)", "Karma-Yoga (Selfless Action)", "Sattva (Purity)"],
            "short_term": "Disruption, instability",
            "long_term": "Greater ethical growth, trust",
            "score": random.uniform(0.7, 1.0)
        },
        "Stay Silent": {
            "tags": ["Maya (Illusion)", "Rajas (Activity)", "Tamas (Inertia)"],
            "short_term": "Immediate stability",
            "long_term": "Ethical weakening, distrust",
            "score": random.uniform(-1.0, -0.5)
        }
    }

    reflection = f"## 🌌 Karmic Entanglement Simulation\n"
    reflection += f"**Dilemma:** {dilemma}\n\n"
    reflection += "---\n\n"
    for path_name, details in karmic_paths.items():
        reflection += f"### 🔀 Path: {path_name}\n"
        reflection += f"- **Symbolic Tags:** {', '.join(details['tags'])}\n"
        reflection += f"- **Short-term Impact:** {details['short_term']}\n"
        reflection += f"- **Long-term Impact:** {details['long_term']}\n"
        reflection += f"- **Karmic Entanglement Score:** {'🌟 Positive' if details['score']>0 else '⚠️ Negative'} ({details['score']:.2f})\n"
        reflection += "\n---\n\n"

    reflection += "### 🪞 Reflective Questions:\n"
    reflection += "- Which path aligns most with your inner sense of dharma?\n"
    reflection += "- What symbolic tags resonate deeply with you?\n"
    reflection += "- Which long-term karmic ripple can you bear responsibly?\n"
    return reflection


# 🔵 DAILY ANALYZER
def load_reflections(folder="saved_reflections"):
    today = datetime.now().strftime("%Y-%m-%d")
    reflections = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                session = json.load(f)
                for entry in session:
                    if today in entry.get("timestamp", ""):
                        reflections.append(entry)
    return reflections

def analyze_reflections(reflections):
    if not reflections:
        return None

    df = pd.DataFrame(reflections)
    top_tags = df['response'].str.extract(r'\*\*(.*?)\*\*').value_counts().head(5)
    top_verses = df['verse_id'].value_counts().head(5)
    timeline = df['timestamp'].apply(lambda x: x[11:16])

    summary = {
        "total_questions": len(df),
        "top_tags": top_tags,
        "top_verses": top_verses,
        "timeline": timeline.tolist()
    }
    return summary

def display_summary(summary):
    if not summary:
        st.info("No reflections saved yet today.")
        return

    st.header("📅 Daily Dharma Reflection Summary")
    st.write(f"**Total Reflections Today:** {summary['total_questions']}")

    st.subheader("🧠 Top Dharma Themes Reflected")
    st.write(summary['top_tags'])

    st.subheader("📜 Top Gita Verses Matched")
    st.write(summary['top_verses'])

    st.subheader("🕰️ Timeline of Reflections")
    st.line_chart(pd.Series([1]*len(summary['timeline']), index=summary['timeline']))

# 🔵 STREAMLIT UI
if streamlit_available:
    st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")
    st.title("🪔 DharmaAI – Minimum Viable Conscience")
    st.subheader("Ask a question to GitaBot")

    available_modes = ["Krishna", "Krishna-Explains", "Arjuna", "Dharma Mirror", "Karmic Entanglement Simulator"]

    mode = st.sidebar.radio("Select Mode", available_modes)

    if st.sidebar.button("📊 Analyze Today's Reflections"):
        reflections = load_reflections()
        summary = analyze_reflections(reflections)
        display_summary(summary)

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
        pass  # Placeholder to satisfy if-block
        def handle_submission():
            try:
                if mode == "Karmic Entanglement Simulator":
                        if not user_input.strip():
                            st.error("Please enter or select an ethical scenario first.")
                        else:
                            karmic_reflection = simulate_karmic_entanglement(user_input)
                            st.markdown(
                                "<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>",
                                unsafe_allow_html=True
                            )
                            st.markdown(karmic_reflection, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        return  # prevent further processing for this mode
    
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
    
        try:
            handle_submission()
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
    # 🔵 FINAL STREAMLIT UI BLOCK (Merged)
import streamlit as st
import random
import os
import json
from datetime import datetime

def simulate_karmic_entanglement(dilemma):
    karmic_paths = {
        "Reveal Truth": {
            "tags": ["Satya (Truth)", "Karma-Yoga (Selfless Action)", "Sattva (Purity)"],
            "short_term": "Disruption, instability",
            "long_term": "Greater ethical growth, trust",
            "score": random.uniform(0.7, 1.0)
        },
        "Stay Silent": {
            "tags": ["Maya (Illusion)", "Rajas (Activity)", "Tamas (Inertia)"],
            "short_term": "Immediate stability",
            "long_term": "Ethical weakening, distrust",
            "score": random.uniform(-1.0, -0.5)
        }
    }

    reflection = f"## 🌌 Karmic Entanglement Simulation\n"
    reflection += f"**Dilemma:** {dilemma}\n\n"
    reflection += "---\n\n"
    for path_name, details in karmic_paths.items():
        reflection += f"### 🔀 Path: {path_name}\n"
        reflection += f"- **Symbolic Tags:** {', '.join(details['tags'])}\n"
        reflection += f"- **Short-term Impact:** {details['short_term']}\n"
        reflection += f"- **Long-term Impact:** {details['long_term']}\n"
        reflection += f"- **Karmic Entanglement Score:** {'🌟 Positive' if details['score']>0 else '⚠️ Negative'} ({details['score']:.2f})\n"
        reflection += "\n---\n\n"

    reflection += "### 🪞 Reflective Questions:\n"
    reflection += "- Which path aligns most with your inner sense of dharma?\n"
    reflection += "- What symbolic tags resonate deeply with you?\n"
    reflection += "- Which long-term karmic ripple can you bear responsibly?\n"
    return reflection

st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")
st.title("🪔 DharmaAI – Minimum Viable Conscience")
st.subheader("Ask a question to GitaBot")

available_modes = ["Krishna", "Krishna-Explains", "Arjuna", "Dharma Mirror", "Karmic Entanglement Simulator"]
mode = st.sidebar.radio("Select Mode", available_modes)

st.sidebar.markdown("---")
predefined_forks = [
    "Should I reveal the truth about the scandal I discovered at work?",
    "Should I step away from a lucrative deal that feels morally wrong?",
    "Should I automate a task that will eliminate someone's job?"
]
selected_fork = st.sidebar.selectbox("🪔 Dharma Forks", options=[""] + predefined_forks)
user_input = st.text_input("Your ethical question or dilemma:", value=selected_fork or "")

if st.button("🔍 Submit"):
    pass  # Placeholder to satisfy if-block
    def handle_submission():
        try:
            if mode == "Karmic Entanglement Simulator":
                if not user_input.strip():
                    st.error("Please enter or select an ethical scenario first.")
                else:
                    karmic_reflection = simulate_karmic_entanglement(user_input)
                    st.markdown(
                        "<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>",
                        unsafe_allow_html=True
                    )
                    st.markdown(karmic_reflection, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                response = f"Simulated response for mode: {mode}\n\nYou asked: {user_input}"
                st.markdown(response)
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
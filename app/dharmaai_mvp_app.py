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

# Define generate_gita_response inline
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

    if mode == "Krishna-GPT":
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
                response = f"**🤖 Krishna-GPT says:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {reply}"
            except Exception as e:
                response = f"❌ Error fetching response from Krishna-GPT: {str(e)}"
        else:
            response = f"**🤖 Krishna-GPT says:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated GPT response here based on dharma logic]'}"

    elif mode == "Krishna-Gemini":
        try:
            import google.generativeai as genai
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                available_models = genai.list_models()
                model_name = next((m.name for m in available_models if "generateContent" in m.supported_generation_methods), "models/gemini-pro")
                model = genai.GenerativeModel(model_name)
                gemini_prompt = f"You are Krishna. Reflect and answer this question with dharmic insight: '{user_input}'. Gita Verse: '{verse_info['Short English Translation']}' tagged '{verse_info['Symbolic Conscience Mapping']}'"
                gemini_reply = model.generate_content(gemini_prompt)
                response = f"**🌟 Krishna-Gemini reflects:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {gemini_reply.text.strip()}"
            else:
                response = f"⚠️ Gemini API key not set.\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated Gemini response]'}"
        except Exception as e:
            response = f"❌ Error fetching response from Krishna-Gemini: {str(e)}"

    elif mode == "Krishna":
        response = f"**🧠 Krishna teaches:**\n\n_You asked:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Symbolic dharma insight would be offered here]'}"

    elif mode == "Arjuna":
        response = (
            f"**😟 Arjuna's Doubt:**\n\n"
            f"> _What should I do about_ **'{user_input}'**?\n\n"
            f"Here are three reflections Arjuna would face:\n"
            f"1. Am I acting from fear or purpose?\n"
            f"2. What attachment makes this choice difficult?\n"
            f"3. If I were not afraid, what would duty ask of me?"
        )

    elif mode == "Vyasa":
        similarity_score = verse_info['similarity'] if verse_info is not None and 'similarity' in verse_info else 'N/A'
        response = (
            f"**📖 Vyasa Narrates:**\n\n"
            f"Long ago, a seeker once asked: _'{user_input}'_.\n\n"
            f"To this, Krishna replied in verse {verse_info['Verse ID'] if verse_info else '[unknown]'}\n"
            f"(Symbolic Tag: {verse_info['Symbolic Conscience Mapping'] if verse_info else '[N/A]'}, Similarity Score: {similarity_score}):\n"
            f"> _{verse_info['Short English Translation'] if verse_info else '[Gita wisdom unavailable]'}_"
        )

    elif mode == "Mirror":
        response = "> You are not here to receive the answer.\n> You are here to see your reflection.\n> Ask again, and you may discover your dharma."

    elif mode == "Dharma Fork Test":
        response = (
            f"🧘 Krishna speaks (via Dharma Fork):\n\n"
            f"**Pursue maximum market share**\n\n"
            f"📜 Dharma: Accelerate access and scale\n"
            f"🌀 Karma: Risk of monopolistic behavior and ethical imbalance\n"
            f"📖 Scroll: When the Wheel is Broken\n"
            f"🔗 Verse: Gita 3.16\n"
            f"🪞 Mirror Protocol: v1.0"
        )

    elif mode == "Technical":
        similarity_score = verse_info['similarity'] if verse_info is not None and 'similarity' in verse_info else 'N/A'
        response = (
            f"🔧 Technical Debug Info:\n"
            f"- Question: {user_input}\n"
            f"- Role: {user_role}\n"
            f"- Matched Verse ID: {verse_info['Verse ID'] if verse_info else 'N/A'}\n"
            f"- Symbolic Tag: {verse_info['Symbolic Conscience Mapping'] if verse_info else 'N/A'}\n"
            f"- Cosine Score: {similarity_score}\n"
            f"- Tokens Used: {total_tokens} (Est. ${estimated_cost})\n"
            f"- Model: {st.session_state.get('OPENAI_MODEL', 'gpt-3.5-turbo')}"
        )

    if streamlit_available:
        st.session_state["Usage Journal"].append({
            "verse_id": verse_info['Verse ID'] if verse_info is not None else None,
            "mode": mode,
            "role": user_role,
            "question": user_input,
            "response": response,
            "tokens": total_tokens,
            "cost_usd": estimated_cost,
            "model": st.session_state.get("OPENAI_MODEL", "gpt-3.5-turbo")
        })

    return response


# Streamlit UI
if streamlit_available:
    st.set_page_config(page_title="DharmaAI MVP", layout="wide")
    st.info(f"🔑 OPENAI API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    st.info(f"🔑 GEMINI API Key: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    st.title("🪔 DharmaAI – Minimum Viable Conscience")
    st.subheader("Ask a question to GitaBot")

    mode = st.sidebar.radio("Select Mode", ["Krishna", "Krishna-GPT", "Krishna-Gemini", "Arjuna", "Vyasa", "Mirror", "Technical", "Dharma Fork Test"])
    user_input = st.text_input("Your ethical question or dilemma:", value="")

    # Load verse matrix
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
        else:
            if streamlit_available:
                st.warning(f"⚠️ Matrix file not found: {path}")

    if st.button("🔍 Submit"):
        if df_matrix is None or df_matrix.empty:
            st.error("🚫 No verse matrix loaded. Please upload or check data file paths.")
            st.stop()
        try:
            response = generate_gita_response(mode, df_matrix=df_matrix, user_input=user_input)
            st.markdown("""
            <div style='border: 1px solid #ddd; padding: 1rem; border-radius: 0.5rem; background-color: #f9f9f9;'>
            """, unsafe_allow_html=True)
            st.markdown(response)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

    if "Usage Journal" in st.session_state and st.session_state["Usage Journal"]:
        with st.expander("🕰️ View Past Interactions"):
            st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))

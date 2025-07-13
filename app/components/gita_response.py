import os
import json
from datetime import datetime
import streamlit as st
from utils.helpers import get_embedding, cosine_similarity
from components.modes import generate_arjuna_reflections, generate_dharma_mirror_reflections

def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or len(user_input.strip()) < 5:
        return "🛑 Please ask a more complete or meaningful question."

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

        if mode == "Krishna-Explains":
            import openai
            if os.getenv("OPENAI_API_KEY"):
                try:
                    system_prompt = f"You are Krishna from the Bhagavad Gita. Provide dharma-aligned, symbolic, and contextual guidance.\nVerse context: '{verse_info['Short English Translation']}' with symbolic tag '{verse_info['Symbolic Conscience Mapping']}'"
                    completion = openai.ChatCompletion.create(
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
            response = f"**🤖 Krishna-Explains (offline mode):**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated GPT response]'}"

if mode == "Krishna":
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

elif mode == "Karmic Entanglement Simulator":
        # Placeholder: simulate karmic entanglement
        response = (
            f"## 🧬 Karmic Entanglement Simulation\n\n"
            f"_Contemplating your question:_ **{user_input}**\n\n"
            f"Two dharmic echoes appear across lives:\n\n"
            f"- In one, attachment leads to repetition.\n"
            f"- In the other, sacrifice liberates the flow.\n\n"
            f"Which karmic fork shall you choose?"
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

elif mode == "Technical":
        similarity_score = verse_info['similarity'] if verse_info is not None and 'similarity' in verse_info else 'N/A'
        response = (
            f"🔧 Technical Debug Info:\n"
            f"- Question: {user_input}\n"
            f"- Mode: {mode}\n"
            f"- Matched Verse ID: {verse_info['Verse ID'] if verse_info else 'N/A'}\n"
            f"- Symbolic Tag: {verse_info['Symbolic Conscience Mapping'] if verse_info else 'N/A'}\n"
            f"- Cosine Score: {similarity_score}\n"
            f"- Model: {st.session_state.get('OPENAI_MODEL', 'gpt-3.5-turbo')}"
        )

elif mode == "Forked Fate Contemplation":
        response = (
            f"## 🧭 Forked Fate Contemplation\n\n"
            f"_You asked:_ **{user_input}**\n\n"
            f"Two futures are unfolding:\n"
            f"1. One bound by karma, obligation, or fear.\n"
            f"2. One born of dharma, courage, or sacrifice.\n\n"
            f"Which fork holds your true self?"
        )
# Save locally
st.session_state["Usage Journal"].append({
        "verse_id": verse_info['Verse ID'] if verse_info is not None else None,
        "mode": mode,
        "question": user_input,
        "response": response,
        "timestamp": datetime.now().isoformat()
    })

    SAVE_FOLDER = os.path.join(os.getcwd(), "saved_reflections")
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    session_filename = os.path.join(SAVE_FOLDER, f"session_{timestamp}.json")
    try:
        with open(session_filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state["Usage Journal"], f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"❌ Failed to save reflection: {e}")

    return response

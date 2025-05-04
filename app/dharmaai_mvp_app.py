import streamlit as st
import sys
import random
import os
import re
import json
from datetime import datetime
import openai
import pandas as pd
import numpy as np
from openai import OpenAI


def main():
    try:
        openai_available = True
    except ImportError:
        openai_available = False

    try:
        streamlit_available = True
    except ImportError:
        streamlit_available = False

    try:
        pd
    except ImportError:
        pd = None

    # MAIN GITA RESPONSE GENERATOR
    def generate_gita_response(mode, df_matrix, user_input=None):
        if not user_input or len(user_input.strip()) < 5:
            pass

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
                    response = f"** Krishna-Explains :**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {reply}"
                except Exception as e:
                    response = f" Error fetching response from Krishna-Explains: {str(e)}"
            else:
                response = f"** Krishna-Explains says:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated GPT response based on dharma logic]'}"

        elif mode == "Krishna":
            response = f"** Krishna teaches:**\n\n_You asked:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Symbolic dharma insight would be offered here]'}"

        # Further logic for other modes continues...

        return response

if __name__ == '__main__':
    main()

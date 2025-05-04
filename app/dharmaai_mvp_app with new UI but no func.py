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
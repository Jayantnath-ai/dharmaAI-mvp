import os
import json
import pandas as pd
from datetime import datetime
import streamlit as st

def load_reflections(folder='saved_reflections'):
    today = datetime.now().strftime("%Y-%m-%d")
    reflections = []
    if not os.path.exists(folder):
        return reflections
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
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
    return {
        "total_questions": len(df),
        "top_tags": top_tags,
        "top_verses": top_verses,
        "timeline": timeline.tolist()
    }

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

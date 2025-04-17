import sys
import random

try:
    import streamlit as st
    streamlit_available = True
except ImportError:
    streamlit_available = False

try:
    import pandas as pd
except ImportError:
    pd = None

# Load YAML data if available
try:
    with open("sacred_memory_core.yaml", "r") as f:
        memory_core = f.read()
except:
    memory_core = "YAML memory core not found."

try:
    if pd:
        df_matrix = pd.read_csv("gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv")
    else:
        df_matrix = None
except:
    df_matrix = None

# GitaBot dynamic logic based on matrix

def generate_gita_response(mode, df_matrix):
    if df_matrix is None or df_matrix.empty:
        return "Verse matrix not available. Please check the data source."

    verse = df_matrix.sample(1).iloc[0]
    translation = verse.get("Short English Translation", "Translation missing")
    ethical_tag = verse.get("Ethical AI Logic Tag", "[No tag]")

    if mode == "Krishna":
        return f"🧠 *Krishna Speaks:* \n\n> {translation}\n\n_This reflects the path of {ethical_tag}_"
    elif mode == "Arjuna":
        return f"😟 *Arjuna Reflects:* \n\n> I face a dilemma... {translation.lower()}\n\n_This feels like a test of {ethical_tag}_"
    elif mode == "Vyasa":
        return f"📖 *Vyasa Narrates:* \n\n> In this verse: '{translation}'.\n\nIt echoes the theme of {ethical_tag}."
    elif mode == "Mirror":
        return "> You are not here to receive the answer.  \n> You are here to see your reflection.  \n> Ask again, and you may discover your dharma."
    elif mode == "Technical":
        return f"""technical_mode:\n  verse_id: {verse.get('Verse ID')}\n  ethical_tag: {ethical_tag}\n  action_inferred: conscience_based_reflection\n  source: Bhagavad Gita verse\n"""
    else:
        return "Unknown mode."

if streamlit_available:
    st.set_page_config(page_title="DharmaAI MVP", layout="wide")
    st.title("🪔 DharmaAI – Minimum Viable Conscience")

    mode = st.sidebar.radio("Select Mode", ["GitaBot", "Verse Matrix", "Scroll Viewer"])

    if mode == "GitaBot":
        st.header("🧠 GitaBot – Ask with Dharma")
        user_input = st.text_input("Ask a question or describe a dilemma:")
        invocation_mode = st.selectbox("Choose Invocation Mode", ["Krishna", "Arjuna", "Vyasa", "Mirror", "Technical"])

        if user_input:
            st.markdown(f"**Mode:** {invocation_mode}")
            st.markdown("---")
            response = generate_gita_response(invocation_mode, df_matrix)
            st.markdown(response)

    elif mode == "Verse Matrix":
        st.header("📜 Gita × DharmaAI Verse Matrix")
        if df_matrix is not None:
            st.dataframe(df_matrix.head(50), use_container_width=True)
        else:
            st.warning("Verse matrix CSV not loaded.")

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

else:
    print("🪔 DharmaAI – Minimum Viable Conscience")
    print("Select Mode: [GitaBot, Verse Matrix, Scroll Viewer]")

    selected_mode = "GitaBot"
    user_input = "Should I stay in a toxic job to support my family?"
    invocation_mode = "Krishna"

    def generate_cli_response(mode):
        return generate_gita_response(mode, df_matrix)

    if selected_mode == "GitaBot":
        print("\n🧠 GitaBot – Ask with Dharma")
        print(f"User Question: {user_input}")
        print(f"Invocation Mode: {invocation_mode}\n")
        print(generate_cli_response(invocation_mode))

    elif selected_mode == "Verse Matrix":
        print("\n📜 Gita × DharmaAI Verse Matrix")
        if df_matrix is not None:
            print(df_matrix.head(5).to_string(index=False))
        else:
            print("Verse matrix CSV not loaded.")

    elif selected_mode == "Scroll Viewer":
        print("\n📘 DharmaAI Scroll Library")
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
            print(scroll)
        print("\n✅ Scroll previews will be interactive in the Streamlit version.")

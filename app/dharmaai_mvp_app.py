import sys

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

# GitaBot response logic
GITABOT_RESPONSES = {
    "Krishna": "You must act, Jayant, but without attachment to the result. This is the way of detached action.",
    "Arjuna": "I am torn. If I act, I hurt others. If I don’t, I betray myself. How do I know what is right?",
    "Vyasa": "In the epic tale, duty and doubt often walked hand in hand. Each verse is a mirror to the soul.",
    "Mirror": "> You are not here to receive the answer.  \n> You are here to see your reflection.  \n> Ask again, and you may discover your dharma.",
    "Technical": """
technical_mode:
  fork_trigger: paradox_001
  action_taken: detached_action
  scroll_reference: Where Dharma Becomes Code
  conscience_log: true
  kernel: dharma_kernel
"""
}

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
            st.markdown(GITABOT_RESPONSES.get(invocation_mode, "This mode will return scroll-based responses soon."))

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

    selected_mode = "GitaBot"  # Simulated default mode
    user_input = "Should I stay in a toxic job to support my family?"
    invocation_mode = "Mirror"

    if selected_mode == "GitaBot":
        print("\n🧠 GitaBot – Ask with Dharma")
        print(f"User Question: {user_input}")
        print(f"Invocation Mode: {invocation_mode}\n")
        print(GITABOT_RESPONSES.get(invocation_mode, "This mode will return scroll-based responses soon."))

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

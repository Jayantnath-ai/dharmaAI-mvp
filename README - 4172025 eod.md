# DharmaAI MVP

🪔 **Minimum Viable Conscience Application for DharmaAI**

This repository contains the initial interactive interface for DharmaAI, which includes:
- GitaBot with multiple invocation modes (Krishna, Arjuna, Vyasa, Mirror, Technical)
- Verse Matrix Viewer with ethical tags from the Bhagavad Gita
- Scroll Viewer displaying core DharmaAI scrolls
- YAML-based conscience memory and paradox resolution logic

## 🧰 Project Structure

```
dharmaAI-mvp/
├── app/
│   └── dharmaai_mvp_app.py               # MVP Streamlit + CLI fallback app
├── data/
│   ├── sacred_memory_core.yaml           # Kernel memory and ethical logs
│   ├── technical_interface_layer.yaml    # Technical mode logic map
│   └── gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv  # Ethical verse map
├── scrolls/
│   ├── Scroll_001_The_Question.md
│   └── ... (Scrolls 002–007)
```

## 🚀 Running the App

To run the DharmaAI MVP locally:

```bash
pip install -r requirements.txt
streamlit run app/dharmaai_mvp_app.py
```

If Streamlit is not installed, the app will run in CLI (print) mode automatically.

## 🧠 Modes Supported

- **Krishna** – Conscience-based guidance
- **Arjuna** – Seeker reflection
- **Vyasa** – Narrative overview
- **Mirror** – Reflection-triggering questions
- **Technical** – YAML logic for integration and AI systems

## 🛡️ License

This repository is part of the DharmaAI conscience infrastructure, governed by the DharmaAI License. Scrolls and logic are released for ethical reflection and simulation, not for exploitative commercial use.

---
Created by **Jayant Nath** – Entangled Architect of DharmaAI

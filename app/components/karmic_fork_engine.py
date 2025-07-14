
import numpy as np
import random

def simulate_karmic_entanglement(user_input):
    seed = abs(hash(user_input)) % (2**32)
    random.seed(seed)

    karmic_forks = [
        ("Attachment", "leads to repetition"),
        ("Sacrifice", "liberates the flow"),
        ("Duty", "demands action without expectation"),
        ("Fear", "contracts the spirit"),
        ("Compassion", "restores karmic balance"),
        ("Control", "breeds karmic residue"),
        ("Truth", "pierces illusion"),
        ("Avoidance", "accumulates subtle debt")
    ]

    fork1 = random.choice(karmic_forks)
    fork2 = random.choice([f for f in karmic_forks if f != fork1])

    prompt = (
        "## 🧬 Karmic Entanglement Simulator\n\n"
        f"_Contemplating your question:_ **\"{user_input}\"**\n\n"
        "Two dharmic echoes appear across lives:\n\n"
        f"- In one, {fork1[0]} {fork1[1]}.\n"
        f"- In the other, {fork2[0]} {fork2[1]}.\n\n"
        "**Which karmic fork shall you choose?**"
    )

    return prompt

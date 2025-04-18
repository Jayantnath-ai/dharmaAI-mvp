
import yaml
from pathlib import Path

def load_simulation_fork(fork_name: str):
    path = Path(f"simulations/{fork_name}.yaml")
    if not path.exists():
        return None
    
    with path.open("r") as f:
        fork_data = yaml.safe_load(f)
    return fork_data

def resolve_dharma_fork_from_yaml(user_query: str, fork_name: str):
    fork = load_simulation_fork(fork_name)
    if not fork:
        return {"error": "Fork not found"}

    # Naive matching based on intent
    if "share" in user_query.lower() or "open" in user_query.lower():
        chosen = fork["options"][1]
    else:
        chosen = fork["options"][0]

    return {
        "ethical_path": chosen["choice"],
        "dharma": chosen["dharma"],
        "karma": chosen["karma"],
        "verse_ref": chosen["verse_ref"],
        "scroll_ref": chosen["scroll_ref"]
    }

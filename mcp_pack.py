import yaml
import subprocess
datetime
from transformers import pipeline

# 1. Load schema (optional validation)
# 2. Summarizer setup
def generate_summary(texts):
    summarizer = pipeline('summarization', model='t5-small')
    joined = "\n\n".join(texts)
    return summarizer(joined, max_length=128, min_length=32)[0]['summary_text']

# 3. Main pack function
def pack_envelope(scroll_ids, modes, user_id, scroll_text_loader):
    commit = subprocess.check_output(['git','rev-parse','HEAD']).strip().decode()
    ts = datetime.datetime.utcnow().isoformat()
    texts = [scroll_text_loader(s) for s in scroll_ids]
    summary = generate_summary(texts)
    env = {
        'scrolls': scroll_ids,
        'modes': modes,
        'user_id': user_id,
        'timestamp': ts,
        'git_commit': commit,
        'summary': summary
    }
    with open(f'context/envelope_{commit[:7]}.yaml', 'w') as f:
        yaml.safe_dump({'envelope': env}, f)

if __name__ == '__main__':
    # Example invocation; replace loader and IDs as needed
    pack_envelope(
        scroll_ids=['001','002'],
        modes=['Krishna'],
        user_id='jayant',
        scroll_text_loader=lambda sid: open(f'scrolls/{sid}.txt').read()
    )
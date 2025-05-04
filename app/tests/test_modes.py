import pandas as pd
from components import modes

sample_df = pd.DataFrame({
    "Short English Translation": ["Act without attachment", "The soul is eternal", "Surrender to the Divine"],
    "Symbolic Conscience Mapping": ["Anasakti", "Atma-Sarvatra", "Bhakti-Yoga"]
})

sample_df['embedding'] = sample_df['Short English Translation'].apply(lambda x: modes.get_embedding(str(x)))

def test_arjuna_reflection_output():
    reflections, verse = modes.generate_arjuna_reflections("What should I do?", sample_df)
    assert isinstance(reflections, list)
    assert len(reflections) == 3
    assert isinstance(verse, str)

def test_dharma_mirror_reflection_output():
    reflections, verse = modes.generate_dharma_mirror_reflections("Who am I serving?", sample_df)
    assert isinstance(reflections, list)
    assert len(reflections) == 3
    assert isinstance(verse, str)

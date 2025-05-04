import pandas as pd
from components import gita_response

sample_df = pd.DataFrame({
    "Short English Translation": ["You have a right to perform your duty", "The wise do not grieve for the living or the dead"],
    "Symbolic Conscience Mapping": ["Karma-Yoga", "Jnana-Yoga"],
    "Verse ID": ["2.47", "2.11"]
})
sample_df['embedding'] = sample_df['Short English Translation'].apply(lambda x: gita_response.get_embedding(str(x)))

def test_generate_gita_response_modes():
    modes = ["Krishna", "Arjuna", "Dharma Mirror", "Forked Fate Contemplation", "Technical", "Vyasa"]
    for mode in modes:
        response = gita_response.generate_gita_response(mode, df_matrix=sample_df, user_input="What is my duty?")
        assert isinstance(response, str)
        assert len(response) > 10

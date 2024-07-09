import pandas as pd
from pysentimiento import create_analyzer
file_path = '/path/to/your/data.csv'
df = pd.read_csv(file_path)
analyzer = create_analyzer(task="sentiment", lang="en")
df['sentiment'] = df['text'].apply(lambda x: analyzer.predict(x).output)
import ace_tools as tools; tools.display_dataframe_to_user(name="Sentiment Analysis Results", dataframe=df)

import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str):
    return re.sub(r'[^\w\s]', '', text.lower())

def load_data(ruta:  str):
    """Carga los datos desde un archivo CSV y asigna valores numéricos a las etiquetas de sentiment."""
    df = pd.read_csv(ruta)
    df["sentiment_num"] = df["sentiment"].map({
        "positivo": 2,
        "neutral": 1,
        "negativo": 0
    })
    
    return df

def split_dataset(df, test_size=0.2, random_state=42):
    return train_test_split(
        df["message"],
        df["sentiment_num"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment_num"]
    )
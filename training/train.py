from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import joblib
import nltk

from preprocess import load_data, split_dataset

nltk.download('stopwords', quiet=True)

DATA_PATH = "data/reviews.csv"
MODEL_PATH = "models/sentiment_model.joblib"

def train():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_dataset(df)

    stop_words_es = stopwords.words("spanish")

    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            stop_words=stop_words_es,
            max_features=8000,
            ngram_range=(1, 2)
        )),
        ("model", LogisticRegression(max_iter=1000)) # Asegurar convergencia
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    train()

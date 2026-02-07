import joblib
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import load_data, split_dataset

DATA_PATH = "data/reviews.csv"
MODEL_PATH = "models/sentiment_model.joblib"

def evaluate():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_dataset(df)

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    print("=== Reporte ===")
    print(classification_report(y_test, y_pred))

    print("=== Matriz de Confusión ===")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()

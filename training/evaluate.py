import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from training.preprocess import load_data, split_dataset

DATA_PATH = "data/reviews.csv"
MODEL_PATH = "models/sentiment_model.joblib"

def evaluate():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_dataset(df)

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    
    y_proba = model.predict_proba(X_test)
    max_confidence = y_proba.max(axis=1)
    
    print("\n=== Reporte de clasificación ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Matriz de confusión ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Métricas de evaluación ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (macro):", precision_score(y_test, y_pred, average="macro"))
    print("Recall (macro):", recall_score(y_test, y_pred, average="macro"))
    print("F1 (macro):", f1_score(y_test, y_pred, average="macro"))
    
    print("\n=== Análisis de confianza ===")
    print("Confianza media:", max_confidence.mean())
    print("Confianza mínima:", max_confidence.min())

if __name__ == "__main__":
    evaluate()

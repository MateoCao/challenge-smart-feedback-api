# The Smart Feedback API

Este repositorio corresponde a un **challenge técnico**, cuyo objetivo es implementar un motor de análisis de sentimiento capaz de clasificar reseñas de texto en categorías (positivo, neutral, negativo) utilizando procesamiento de lenguaje natural (NLP).
Incluye un pipeline de entrenamiento con **Scikit-Learn** y una API de consulta construida con **FastAPI**.

## Enfoque del modelo
El modelo utiliza:
- Vectorización **TF-IDF** (unigramas y bigramas)
- Clasificación mediante **Logistic Regression**


El dataset se encuentra balanceado entre las tres clases y el pipeline fue diseñado para evitar *data leakage* mediante el uso de `Pipeline` de Scikit-Learn.

## Tecnologías
- Python
- Scikit-Learn
- FastAPI

## Instalación

Se recomienda crear un entorno virtual para mantener las dependencias aisladas.
1) Clonar repositorio.

```bash
git clone "https://github.com/MateoCao/challenge-smart-feedback-api"
```
2) Crear entorno virtual.
```bash
cd challenge-smart-feedback-api
python -m venv venv
```
3) Activar entorno.

-  En windows:
```bash
venv\Scripts\activate
```
- En Linux:

```bash
source venv/bin/activate
```
4) Instalar dependencias.
```bash
pip install -r requirements.txt
```

## Uso

El repositorio incluye el modelo preentrenado en la carpeta /models. Si se desea reentrenar el modelo desde cero ejecutar:

```bash
python training/train.py
```
Esto generará el archivo .joblib del modelo entrenado en /models.

Para evaluar el rendimiento del modelo sobre el conjunto de test:

```bash
python training/evaluate.py
```

Esto mostrará en consola el **Classification Report** y la **Confusion Matrix**.

## API (FastAPI)
La API REST se encuentra actualmente en desarrollo. Su objetivo es exponer el modelo entrenado a través de un endpoint que recibe uno o más textos en formato JSON y devuelve:
- Sentiment (positivo / neutral / negativo)
- Score de confianza asociado a la predicción
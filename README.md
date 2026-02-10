# The Smart Feedback API

Este repositorio corresponde a un **challenge técnico**, cuyo objetivo es implementar un motor de análisis de sentimiento capaz de clasificar reseñas de texto en categorías (positivo, neutral, negativo) utilizando procesamiento de lenguaje natural (NLP).
Incluye un pipeline de entrenamiento con **Scikit-Learn** y una API de consulta construida con **FastAPI**.

## Enfoque del modelo
El modelo utiliza:
- Vectorización **TF-IDF** (unigramas y bigramas)
- Clasificación mediante **Logistic Regression**

Para este challenge se tomaron decisiones orientadas a la Escabilidad e Ingeniería de Software:

- Se implementó un pipeline para evitar *data leakage* mediante el uso de `Pipeline` de Scikit-Learn. En este se integró la función de limpieza de texto (`clean_text`) para realizar el preprocesamiento internamente.
- Se redujeron las características de ~77,000 (valor por defecto) a 1,500 con max_features, aplicando además un min_df=5 con el objetivo de disminuir el overfitting. Además, se configuró `ngram_range=(1, 2)` para permitir que el modelo aprenda combinaciones de dos palabras.
- La clase `SentimentModel` en `app/model.py` implementa un patrón donde el modelo se carga en memoria una sola vez al iniciar el servidor, evitando lecturas de disco innecesarias en cada petición.

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

El repositorio incluye el modelo preentrenado en la carpeta /models. Si se desea reentrenar el modelo desde cero ejecutar desde la carpeta root:

```bash
python -m training.train
```
Esto generará el archivo .joblib del modelo entrenado en /models. También se descargaran las stopwords de NLTK necesarias para el preprocesamiento. 

Para evaluar el rendimiento del modelo sobre el conjunto de test:

```bash
python -m training.evaluate
```

Esto mostrará en consola el **Classification Report** y la **Confusion Matrix** junto con otras métricas de evaluación y un análisis de confianza media y mínima.

## API (FastAPI)
La API tiene como objetivo exponer el modelo entrenado a través de un endpoint que recibe un texto en formato JSON.
### Ejemplo de consulta
El endpoint recibe un objeto JSON con el texto a analizar y devuelve el sentimiento junto con el score de confianza.
#### Request
```bash
{
  "text": "Imposible trabajar así. La plataforma es extremadamente inestable y el soporte técnico brilla por su ausencia, devolviendo respuestas predefinidas que no resuelven nada. El sistema se queda colgado constantemente, lo que nos hace perder horas de trabajo cada semana. Es una situación crítica y, de no solucionarse hoy mismo, empezaremos la migración a otra herramienta."
}
```
#### Response
```bash
{
  "text": "Imposible trabajar así. La plataforma es extremadamente inestable y el soporte técnico brilla por su ausencia, devolviendo respuestas predefinidas que no resuelven nada. El sistema se queda colgado constantemente, lo que nos hace perder horas de trabajo cada semana. Es una situación crítica y, de no solucionarse hoy mismo, empezaremos la migración a otra herramienta.",
  "sentiment": "negativo",
  "score": 0.7625693684481034
}
```